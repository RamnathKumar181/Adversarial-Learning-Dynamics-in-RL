from garage.envs import GymEnv


def get_env_from_gym(env_name):
    return GymEnv(env_name)
