#!/usr/bin/env python3
"""This is an example to train Task Embedding PPO with PointEnv."""
# pylint: disable=no-value-for-parameter
import tensorflow as tf
from garage import wrap_experiment
import metaworld
from garage.experiment import MetaWorldTaskSampler
from garage.envs import normalize

from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from src.algos.ate_ppo import ATEPPO
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy
from garage.trainer import TFTrainer
import wandb
import random




'''
['assembly-v2', 'basketball-v2', 'bin-picking-v2',
'box-close-v2', 'button-press-topdown-v2',
'button-press-topdown-wall-v2', 'button-press-v2',
'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2',
'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2',
'door-close-v2', 'door-lock-v2', 'door-open-v2',
'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2',
'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2',
'hammer-v2', 'handle-press-side-v2', 'handle-press-v2',
'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2',
'peg-insert-side-v2', 'pick-place-wall-v2',
'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
'push-v2', 'pick-place-v2', 'plate-slide-v2',
'plate-slide-side-v2', 'plate-slide-back-v2',
'plate-slide-back-side-v2', 'peg-unplug-side-v2',
'soccer-v2', 'stick-push-v2', 'stick-pull-v2',
'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2',
'sweep-into-v2', 'sweep-v2', 'window-open-v2',
'window-close-v2']
'''


'''
stick-push-v2, push-wall-v2, push-v2
'''

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


    env_set = ['push-v2', 'stick-push-v2', 'push-wall-v2', 'coffee-push-v2']
    envs = []
    for env_name in env_set:
        env = metaworld.MT1(env_name).train_classes[env_name]()
        task = random.choice(metaworld.MT1(env_name).train_tasks)
        env.set_task(task)
        envs.append(env)

    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')
    latent_length = 4
    inference_window = 6
    batch_size = 5000 * len(envs)
    policy_ent_coeff = 2e-2
    encoder_ent_coeff = 2e-4
    inference_ce_coeff = 5e-2
    embedding_init_std = 0.1
    embedding_max_std = 0.2
    embedding_min_std = 1e-6
    policy_init_std = 1.0
    policy_max_std = 1.5
    policy_min_std = 0.5

    with TFTrainer(snapshot_config=ctxt) as trainer:
        task_embed_spec = ATEPPO.get_encoder_spec(env.task_space,
                                                  latent_dim=latent_length)

        task_encoder = GaussianMLPEncoder(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 10),
            std_share_network=True,
            init_std=embedding_init_std,
            max_std=embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            std_output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        traj_embed_spec = ATEPPO.get_infer_spec(
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
            hidden_sizes=(32, 32),
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

        algo = ATEPPO(env_spec=env.spec,
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
                      encoder_optimizer_args=dict(
                          batch_size=64,
                          max_optimization_epochs=10,
                          learning_rate=1e-3,
                      ),
                      policy_optimizer_args=dict(
                          batch_size=64,
                          max_optimization_epochs=10,
                          learning_rate=1e-3,
                      ),
                      inference_optimizer_args=dict(
                          batch_size=64,
                          max_optimization_epochs=10,
                          learning_rate=1e-3,
                      ),
                      center_adv=True,
                      stop_ce_gradient=True,
                      num_embedding_itr=1,
                      num_policy_itr=10,
                      num_inference_itr=5)

        trainer.setup(algo, env)
        trainer.train(n_epochs=4000, batch_size=batch_size, plot=False)


def train_ate_ppo_mt1(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
