#!/usr/bin/env python3
"""This is an example to train Task Embedding and Adversarial Task Embedding PPO with MT1 (Domain adaptation)."""
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
import joblib


@wrap_experiment
def train(ctxt):
    """Train Task Embedding and Adversarial Task Embedding PPO with MT1 (Domain adaptation).

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_epochs (int): Total number of epochs for training.
        batch_size_per_task (int): Batch size of samples for each task.

    """
    set_seed(config.seed)

    mt1 = metaworld.MT1(config.mt1_env_name)
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
    batch_size = 5000 * len(envs)
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
        experiment = joblib.load(config.folder)
        pre_trained_policy = experiment['algo'].policy
        pre_trained_inference = experiment['algo']._inference
        print('pre_trained_policy', pre_trained_policy)
        print('pre_trained_inference', pre_trained_inference)

        task_embed_spec = TEPPO.get_encoder_spec(env.task_space,
                                                 latent_dim=latent_length)

        task_encoder = GaussianMLPEncoder(
            name='new_embedding',
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
            name='new_inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=0.1,
            output_nonlinearity=tf.nn.tanh,
            std_output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        policy = GaussianMLPTaskEmbeddingPolicy(
            name='new_policy',
            env_spec=env.spec,
            encoder=task_encoder,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=policy_max_std,
            init_std=policy_init_std,
            min_std=policy_min_std,
        )

        print("Loading previous parameters")
        inference.set_param_values(pre_trained_inference.get_param_values())
        print("Finished loading Inference parameters")
        print("Using TE-PPO architecture")

        source_params = pre_trained_policy.get_params()
        source_weights = tf.compat.v1.get_default_session().run(source_params)

        target_params = policy.get_params()
        target_weights = tf.compat.v1.get_default_session().run(target_params)

        layer = 0
        while layer < len(target_params):
            print(
                f"Layer {layer+1}: {source_params[layer].name}, {target_params[layer].name}")
            if source_params[layer].shape == target_params[layer].shape:
                target_params[layer].load(source_weights[layer])
                print(f"Loaded weights for layer: {layer+1}")
            layer += 1
        print("Old weights:")
        print(target_weights[-1])
        print("New weights:")
        print(tf.compat.v1.get_default_session().run(policy.get_params())[-1])
        print("Expected new weights:")
        print(source_weights[-1])

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
                         batch_size=256,
                         max_optimization_epochs=10,
                         learning_rate=1e-3,
                     ),
                     inference_optimizer_args=dict(
                         batch_size=256,
                         max_optimization_epochs=10,
                         learning_rate=1e-3,
                     ),
                     center_adv=True,
                     stop_ce_gradient=True,
                     name="test")

        trainer.setup(algo, env)
        trainer.train(n_epochs=10, batch_size=batch_size, plot=False)


def train_ppo_mt1(args):
    global config
    config = args
    train({'log_dir': args.snapshot_dir,
           'use_existing_dir': True})
