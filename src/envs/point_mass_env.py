import numpy as np
from garage.envs import PointEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy


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


def get_point_mass_env(env_args):
    goals = circle(3.0, env_args.n)
    tasks = {
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
    task_names = sorted(tasks.keys())
    task_args = [tasks[t]['args'] for t in task_names]
    task_kwargs = [tasks[t]['kwargs'] for t in task_names]
    task_envs = [
        PointEnv(*t_args, **t_kwargs, max_episode_length=100)
        for t_args, t_kwargs in zip(task_args, task_kwargs)
    ]
    env = MultiEnvWrapper(task_envs, round_robin_strategy, mode='vanilla')
    return env
