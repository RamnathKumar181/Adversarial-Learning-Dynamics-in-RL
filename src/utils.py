from collections import namedtuple
from src.envs import get_point_mass_env, get_env_from_gym
from src.algos import get_te_ppo_algo, get_dqn_algo, get_ate_ppo_algo
Benchmark = namedtuple('Benchmark', 'algo env ')


def get_benchmark_by_name(algo_name, algo_args, env_name, env_args):
    if env_name == "point_mass":
        env = get_point_mass_env(env_args)
    elif env_name in ['CartPole-v0']:
        env = get_env_from_gym(env_name)
    if algo_name == "te_ppo":
        algo = get_te_ppo_algo(algo_args, env)
    elif algo_name == "dqn":
        algo = get_dqn_algo(algo_args, env)
    elif algo_name == "ate_ppo":
        algo = get_ate_ppo_algo(algo_args, env)
    return Benchmark(algo=algo,
                     env=env)
