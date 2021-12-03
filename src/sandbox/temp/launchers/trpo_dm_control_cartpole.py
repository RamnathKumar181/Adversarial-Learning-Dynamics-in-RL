from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.policies import GaussianMLPPolicy
from garage.misc.instrument import stub
from garage.misc.instrument import run_experiment

from sandbox.embed2learn.envs import DmControlEnv


def run_task(*_):
    env = normalize(
        DmControlEnv(
            domain_name='cartpole', task_name='balance',
            visualize_reward=True))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=400,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
