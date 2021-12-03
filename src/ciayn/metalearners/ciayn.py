import torch
import numpy as np
import torch.nn as nn
from src.maml_rl.metalearners.base import GradientBasedMetaLearner
from src.utils.torch_utils import to_numpy
from src.utils.reinforcement_learning import reinforce_loss_updated
from src.utils.optimization import JSD
from garage.tf.spaces import Box

# Embedding params
LATENT_LENGTH = 32
TRAJ_ENC_WINDOW = 16


class CIAYN(GradientBasedMetaLearner):
    def __init__(self,
                 env,
                 policy,
                 encoder,
                 inference,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(CIAYN, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.embedding_network = encoder
        self.inference_network = inference

    def get_one_hot(self, t):
        """ Get the one-hot distribution for a task """
        onehot = np.zeros(self.policy.task_space.shape, dtype=np.float32)
        onehot[t] = 1
        return onehot

    def step(self,
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0])
        logs = {}
        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        logs['loss_before'] = to_numpy(old_losses)
        old_loss = sum(old_losses) / num_tasks

        for task in num_tasks:
            for _ in enumerate(self.max_steps):
                self.train_network(train_futures, valid_futures, task, network='encoder')
                self.train_network(train_futures, valid_futures, task, network='policy')
                self.train_network(train_futures, valid_futures, task, network='inference')

                losses, kls, _ = self._async_gather([
                    self.surrogate_loss(train, valid, old_pi=old_pi)
                    for (train, valid, old_pi)
                    in zip(zip(*train_futures), valid_futures, old_pis)])

                improve = (sum(losses) / num_tasks) - old_loss
                if (improve.item() < 0.0):
                    logs['loss_after'] = to_numpy(losses)
        return logs

    def get_loss(self, embedding, predicted_z, pi, task, network):
        loss = torch.tensor(0.0)
        if network == 'encoder':
            loss -= self.alpha*JSD(embedding, task)
            loss += reinforce_loss_updated(self.policy, embedding)
        elif network == 'policy':
            loss += reinforce_loss_updated(self.policy, embedding)
        else:
            loss += nn.CrossEntropyLoss(predicted_z, embedding)
        return loss

    def reset_opt(self):
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.inference_optimizer.zero_grad()

    def opt_step(self, network):
        if network == 'encoder':
            self.encoder_optimizer.step()
        elif network == 'policy':
            self.policy_optimizer.step()
        elif network == 'inference':
            self.inference_optimizer.step()

    def train_network(self, train, valid, task, max=False, network='policy'):
        self.embedding_network.eval()
        self.policy.eval()
        self.inference_network.eval()
        if network == 'encoder':
            self.embedding_network.train()
            num_iters = self.encoder_iters
        elif network == 'policy':
            self.policy.train()
            num_iters = self.agent_iters
        elif network == 'inference':
            self.inference_network.train()
            num_iters = self.inference_iters
        one_hot_task = self.get_one_hot(task)
        for _ in enumerate(num_iters):
            embedding = self.embedding_network(one_hot_task)
            pi = self.policy(train[task].observations, embedding)
            act_obs = np.concatenate([self.env.actions, self.env.observations])
            history = np.stack([act_obs] * TRAJ_ENC_WINDOW)
            predicted_z = self.inference_network(history)
            loss = self.get_loss(embedding, predicted_z, pi, one_hot_task, network)
            if max:
                loss = -loss
            self.reset_opt()
            loss.backward()
            self.opt_step(network)
