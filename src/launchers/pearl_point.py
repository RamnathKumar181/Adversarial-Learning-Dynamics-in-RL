#!/usr/bin/env python3
"""This is an example to train Task Embedding PPO with PointEnv."""
# pylint: disable=no-value-for-parameter
import numpy as np
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import PointEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from src.algos.te_ppo import TEPPO
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy
from garage.trainer import TFTrainer
import wandb


def circle(r, n):
    """Generate n points on a circle of radius r.

    Args:
        r (float): Radius of the circle.
        n (int): Number of points to generate.

    Yields:
        tuple(float, float): Coordinate of a point.

    """
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


N = 4
goals = circle(3.0, N)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'never_done': False,
            'done_bonus': 10.0,
        }
    }
    for i, g in enumerate(goals)
}


@wrap_experiment
def train(ctxt):
    """Train Task Embedding PPO with PointEnv.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_epochs (int): Total number of epochs for training.
        batch_size_per_task (int): Batch size of samples for each task.

    """
    set_seed(1)
    tasks = TASKS
    batch_size = 1024 * len(TASKS)
    trainer = Trainer(ctxt)

    task_names = sorted(tasks.keys())
    task_args = [tasks[t]['args'] for t in task_names]
    task_kwargs = [tasks[t]['kwargs'] for t in task_names]
    task_envs = [
        PointEnv(*t_args, **t_kwargs, max_episode_length=100)
        for t_args, t_kwargs in zip(task_args, task_kwargs)
    ]
    env = MultiEnvWrapper(task_envs, round_robin_strategy, mode='vanilla')

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    sampler = LocalSampler(agents=None,
                           envs=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    trainer.setup(algo, env)
    trainer.train(n_epochs=config.epochs, batch_size=batch_size, plot=False)


def train_te_ppo_pointenv(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
    print(args)
