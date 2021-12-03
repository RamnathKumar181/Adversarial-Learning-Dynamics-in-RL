import logging
import torch
import wandb
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import trange
import numpy as np
from src.maml_rl.metalearners import CIAYN
from src.maml_rl.baseline import LinearFeatureBaseline
from src.samplers import MultiTaskSampler
from src.utils.helpers import get_policy_for_env, get_input_size, get_encoder, get_inference
from src.utils.reinforcement_learning import get_returns


class CIAYNTrainer():
    """
    CIAYN Trainer
    """

    def __init__(self, config):
        self.config = config
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Configuration while training: {self.config}")
        self._build()

    def _build(self):
        self._get_env()
        self._get_models()
        self._get_baseline()
        self._get_metalearner()
        self._train()

    def _train(self):
        num_iterations = 0
        for batch in trange(self.config['num-batches']):
            tasks = self.sampler.sample_tasks(num_tasks=self.config['meta-batch-size'])
            futures = self.sampler.sample_async(tasks,
                                                num_steps=self.config['num-steps'],
                                                fast_lr=self.config['fast-lr'],
                                                gamma=self.config['gamma'],
                                                gae_lambda=self.config['gae-lambda'],
                                                device=self.config['device'])
            logs = self.metalearner.step(*futures,
                                         max_kl=self.config['max-kl'],
                                         cg_iters=self.config['cg-iters'],
                                         cg_damping=self.config['cg-damping'],
                                         ls_max_steps=self.config['ls-max-steps'],
                                         ls_backtrack_ratio=self.config['ls-backtrack-ratio'])

            train_episodes, valid_episodes = self.sampler.sample_wait(futures)
            num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
            num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
            valid_returns = get_returns(valid_episodes)
            logs.update(tasks=tasks,
                        num_iterations=num_iterations,
                        train_returns=get_returns(train_episodes[0]),
                        valid_returns=valid_returns)
            print(valid_returns, type(valid_returns))
            wandb.log({"Rewards": valid_returns})

        # Save policy
        if self.config['output_folder'] is not None:
            with open(self.config['policy_filename'], 'wb') as f:
                torch.save(self.policy.state_dict(), f)

    def _get_env(self):
        # Environment
        self.env = gym.make(self.config['env-name'], **self.config.get('env-kwargs', {}))
        self.env.close()

    def _get_models(self):
        # Policy
        self.policy = get_policy_for_env(self.env,
                                         hidden_sizes=self.config['hidden-sizes'],
                                         nonlinearity=self.config['nonlinearity'])
        self.policy.share_memory()
        wandb.watch(self.policy)

        # Embedding Network
        self.encoder = get_encoder(hidden_sizes=self.config['encoder-hidden-sizes'],
                                   nonlinearity=self.config['encoder-nonlinearity'])
        wandb.watch(self.encoder)

        # Inference Network
        self.inference = get_inference(hidden_sizes=self.config['inference-hidden-sizes'],
                                       nonlinearity=self.config['inference-nonlinearity'])
        wandb.watch(self.inference)

    def _get_baseline(self):
        # Baseline
        self.baseline = LinearFeatureBaseline(get_input_size(self.env))

    def _get_metalearner(self):
        # Metalearner
        self.metalearner = CIAYN(self.env,
                                 self.policy,
                                 self.encoder,
                                 self.inference,
                                 device=self.config['device'])


class CIAYNTester():
    """
    CIAYN Tester
    """

    def __init__(self, config):
        self.config = config
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Configuration while testing: {self.config}")
        self._build()

    def _build(self):
        self._get_env()
        self._get_policy()
        self._get_baseline()
        self._get_sampler()
        self._test()
        if self.config['plot']:
            self.plot_world()

    def plot_world(self):
        self.env = gym.make(self.config['env-name'], **self.config.get('env-kwargs', {}))
        video = VideoRecorder(self.env, self.config["plot_name"])
        observations = self.env.reset()
        self.env.render()
        video.capture_frame()
        with torch.no_grad():
            while not self.envs.dones.all():
                observations_tensor = torch.from_numpy(observations)
                pi = self.policy(observations_tensor)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, _, infos = self.envs.step(actions)
                self.env.render()
                video.capture_frame()
                observations = new_observations
        self.env.close()
        video.close()

    def _test(self):
        logs = {'tasks': []}
        train_returns, valid_returns = [], []
        for batch in trange(self.config['num-batches-test']):
            tasks = self.sampler.sample_tasks(num_tasks=self.config['meta-batch-size-test'])
            train_episodes, valid_episodes = self.sampler.sample(tasks,
                                                                 num_steps=self.config['num-steps'],
                                                                 fast_lr=self.config['fast-lr'],
                                                                 gamma=self.config['gamma'],
                                                                 gae_lambda=self.config['gae-lambda'],
                                                                 device=self.config['device'])

            logs['tasks'].extend(tasks)
            train_returns.append(get_returns(train_episodes[0]))
            valid_returns.append(get_returns(valid_episodes))

        logs['train_returns'] = np.concatenate(train_returns, axis=0)
        logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

        with open(f"{self.config['output_folder']}/results.npz", 'wb') as f:
            np.savez(f, **logs)

    def _get_env(self):
        # Environment
        self.env = gym.make(self.config['env-name'], **self.config.get('env-kwargs', {}))
        self.env.close()

    def _get_policy(self):
        # Policy
        self.policy = get_policy_for_env(self.env,
                                         hidden_sizes=self.config['hidden-sizes'],
                                         nonlinearity=self.config['nonlinearity'])
        with open(self.config['policy_filename'], 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device(self.config['device']))
            self.policy.load_state_dict(state_dict)
        self.policy.share_memory()

    def _get_baseline(self):
        # Baseline
        self.baseline = LinearFeatureBaseline(get_input_size(self.env))

    def _get_sampler(self):
        # Sampler
        self.sampler = MultiTaskSampler(self.config['env-name'],
                                        env_kwargs=self.config.get('env-kwargs', {}),
                                        batch_size=self.config['fast-batch-size'],
                                        policy=self.policy,
                                        baseline=self.baseline,
                                        env=self.env,
                                        seed=self.config['seed'],
                                        num_workers=self.config['num_workers'])
