import torch
import numpy as np

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from src.maml_rl.metalearners.base import GradientBasedMetaLearner
from src.utils.torch_utils import (weighted_mean, detach_distribution,
                                   to_numpy, vector_to_parameters)
from src.utils.optimization import conjugate_gradient
from src.utils.reinforcement_learning import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].
    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).
    fast_lr : float
        Step-size for the inner loop update/fast adaptation.
    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.
    first_order : bool
        If `True`, then the first order approximation of MAML is applied.
    device : str ("cpu" or "cuda")
        Name of the device for the optimization.
    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_futures, first_order=None):
        if first_order is None:
            first_order = self.first_order
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            inner_loss = reinforce_loss(self.policy,
                                        await futures,
                                        params=params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def get_shannon_entropy(self, A, mode="auto"):
        """
        https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
        """
        A = np.asarray(A)

        # Determine distribution type
        if mode == "auto":
            condition = np.all(A.astype(float) == A.astype(int))
            if condition:
                mode = "discrete"
            else:
                mode = "continuous"
        # Compute shannon entropy
        pA = A / A.sum()
        # Remove zeros
        pA = pA[np.nonzero(pA)[0]]
        if mode == "continuous":
            return -np.sum(pA*np.log2(A))
        if mode == "discrete":
            return -np.sum(pA*np.log2(pA))

    def get_mutual_information(self, x, y, mode="auto", normalized=False):
        # Determine distribution type
        if mode == "auto":
            condition_1 = np.all(x.astype(float) == x.astype(int))
            condition_2 = np.all(y.astype(float) == y.astype(int))
            if all([condition_1, condition_2]):
                mode = "discrete"
            else:
                mode = "continuous"

        H_x = self.shannon_entropy(x, mode=mode)
        H_y = self.shannon_entropy(y, mode=mode)
        H_xy = self.shannon_entropy(np.concatenate([x, y]), mode=mode)

        # Mutual Information
        I_xy = H_x + H_y - H_xy
        if normalized:
            return I_xy/np.sqrt(H_x*H_y)
        else:
            return I_xy

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        params = await self.adapt(train_futures,
                                  first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi

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
                self.train_embedding_network(train_futures[task])
                self.train_agent_network(train_futures[task])
                self.train_inference_network(train_futures[task])

                losses, kls, _ = self._async_gather([
                    self.surrogate_loss(train, valid, old_pi=old_pi)
                    for (train, valid, old_pi)
                    in zip(zip(*train_futures), valid_futures, old_pis)])

                improve = (sum(losses) / num_tasks) - old_loss
                if (improve.item() < 0.0):
                    logs['loss_after'] = to_numpy(losses)
        return logs

    def train_embedding_network(self, train_features):
        self.embedding_network.train()
        self.agent_network.eval()
        self.inference_network.eval()
        for _ in enumerate(self.embedding_iters):
            self.train(train_features, max=True)

    def train_agent_network(self, train_features):
        self.embedding_network.eval()
        self.agent_network.train()
        self.inference_network.eval()
        for _ in enumerate(self.agent_iters):
            self.train(train_features)

    def train_inference_network(self, train_features):
        self.embedding_network.eval()
        self.agent_network.eval()
        self.inference_network.train()
        for _ in enumerate(self.inference_iters):
            self.train(train_features)

    def train(self, input, max=False):
        embedding = self.embedding_network(input)
        output = self.agent_network(embedding)
        loss = self.get_loss(embedding, output)
        if max:
            loss = -loss
        loss.backward()
        self.optimizer.step()
