#!/usr/bin/env python3
"""This is an example to train Task Embedding PPO with PointEnv."""
# pylint: disable=no-value-for-parameter
import tensorflow as tf
import metaworld
from garage.experiment import MetaWorldTaskSampler
from garage.envs import normalize

from garage import wrap_experiment
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


def get_env(choice):
    env_enum = {0: 'push-v2',
                1: 'window-open-v2',
                2: 'window-close-v2',
                3: 'drawer-close-v2',
                4: 'drawer-open-v2', }
    return env_enum[choice]


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
    set_seed(config.seed)

    env_name = get_env(config.mt1_env_num)

    mt1 = metaworld.MT1(env_name)
    task_sampler = MetaWorldTaskSampler(mt1,
                                        'train',
                                        lambda env, _: normalize(env),
                                        add_env_onehot=False)
    num_tasks = 50
    envs = [env_up() for env_up in task_sampler.sample(num_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    latent_length = 4
    inference_window = 6
    batch_size = 2000 * len(envs)
    policy_ent_coeff = 2e-2
    encoder_ent_coeff = 2e-2
    inference_ce_coeff = 5e-2
    embedding_init_std = 0.1
    embedding_max_std = 0.2
    embedding_min_std = 1e-6
    policy_init_std = 1.0
    policy_max_std = 1.5
    policy_min_std = 0.5

    with TFTrainer(snapshot_config=ctxt) as trainer:
        task_embed_spec = TEPPO.get_encoder_spec(env.task_space,
                                                 latent_dim=latent_length)

        task_encoder = GaussianMLPEncoder(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=embedding_init_std,
            max_std=embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            std_output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        traj_embed_spec = TEPPO.get_infer_spec(
            env.spec,
            latent_dim=latent_length,
            inference_window_size=inference_window)

        inference = GaussianMLPEncoder(
            name='inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),
            std_share_network=True,
            init_std=0.1,
            output_nonlinearity=tf.nn.tanh,
            std_output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        policy = GaussianMLPTaskEmbeddingPolicy(
            name='policy',
            env_spec=env.spec,
            encoder=task_encoder,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=policy_max_std,
            init_std=policy_init_std,
            min_std=policy_min_std,
        )

        baseline = LinearMultiFeatureBaseline(
            env_spec=env.spec, features=['observations', 'tasks', 'latents'])

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=TaskEmbeddingWorker)

        algo = TEPPO(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     sampler=sampler,
                     inference=inference,
                     discount=0.99,
                     lr_clip_range=0.2,
                     policy_ent_coeff=policy_ent_coeff,
                     encoder_ent_coeff=encoder_ent_coeff,
                     inference_ce_coeff=inference_ce_coeff,
                     use_softplus_entropy=True,
                     optimizer_args=dict(
                         batch_size=32,
                         max_optimization_epochs=10,
                         learning_rate=config.policy_optimizer_lr,
                     ),
                     inference_optimizer_args=dict(
                         batch_size=64,
                         max_optimization_epochs=10,
                         learning_rate=config.inference_optimizer_lr,
                     ),
                     center_adv=True,
                     stop_ce_gradient=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=500, batch_size=batch_size, plot=True)


def train_te_ppo_mt1(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
