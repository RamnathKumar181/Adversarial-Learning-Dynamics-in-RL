import tensorflow as tf
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TEPPO
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy


def get_te_ppo_algo(algo_args, env):
    task_embed_spec = TEPPO.get_encoder_spec(env.task_space,
                                             latent_dim=algo_args.latent_length)

    task_encoder = GaussianMLPEncoder(
        name='embedding',
        embedding_spec=task_embed_spec,
        hidden_sizes=(20, 20),
        std_share_network=True,
        init_std=algo_args.embedding_init_std,
        max_std=algo_args.embedding_max_std,
        output_nonlinearity=tf.nn.tanh,
        std_output_nonlinearity=tf.nn.tanh,
        min_std=algo_args.embedding_min_std,
    )

    traj_embed_spec = TEPPO.get_infer_spec(
        env.spec,
        latent_dim=algo_args.latent_length,
        inference_window_size=algo_args.inference_window)

    inference = GaussianMLPEncoder(
        name='inference',
        embedding_spec=traj_embed_spec,
        hidden_sizes=(20, 20),
        std_share_network=True,
        init_std=0.1,
        output_nonlinearity=tf.nn.tanh,
        std_output_nonlinearity=tf.nn.tanh,
        min_std=algo_args.embedding_min_std,
    )

    policy = GaussianMLPTaskEmbeddingPolicy(
        name='policy',
        env_spec=env.spec,
        encoder=task_encoder,
        hidden_sizes=(32, 16),
        std_share_network=True,
        max_std=algo_args.policy_max_std,
        init_std=algo_args.policy_init_std,
        min_std=algo_args.policy_min_std,
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
                 policy_ent_coeff=algo_args.policy_ent_coeff,
                 encoder_ent_coeff=algo_args.encoder_ent_coeff,
                 inference_ce_coeff=algo_args.inference_ce_coeff,
                 use_softplus_entropy=True,
                 optimizer_args=dict(
                     batch_size=32,
                     max_optimization_epochs=10,
                     learning_rate=1e-3,
                 ),
                 inference_optimizer_args=dict(
                     batch_size=32,
                     max_optimization_epochs=10,
                     learning_rate=1e-3,
                 ),
                 center_adv=True,
                 stop_ce_gradient=True)
    return algo
