from src.launchers import train_te_ppo_pointenv, train_ate_ppo_pointenv
from src.launchers import train_te_ppo_mt10, train_ate_ppo_mt10
from gym import wrappers


def visualize_algorithm_as_video(env, policy):
    env.spec.id = 1
    env = wrappers.Monitor(env, "./gym-results", force=True)
    obs = env.reset()
    for i in range(1000):
        action, _ = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    print("done at step %i" % i)
    env.close()


def get_benchmark_by_name(algo_name, env_name):
    if algo_name == "te_ppo":
        if env_name == "point_mass":
            algo = train_te_ppo_pointenv
        if env_name == "mt10":
            algo = train_te_ppo_mt10
    elif algo_name == "ate_ppo":
        if env_name == "point_mass":
            algo = train_ate_ppo_pointenv
        if env_name == "mt10":
            algo = train_ate_ppo_mt10
    return algo
