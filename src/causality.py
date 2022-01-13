import numpy as np
import copy
import torch
import torch.autograd as autograd
import tensorflow as tf


def get_rewards(env,
                agent,
                z,
                max_path_length=np.inf,
                animated=False,
                speedup=1,
                always_return_paths=False):

    rewards = 0
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    while episode_length < (max_path_length or np.inf):
        a, agent_info = agent.get_action_given_latent(last_obs, z)
        es = env.step(a)
        rewards += es.reward
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation
    return rewards


def plot_ace(mu, std, num_c, policy, env, num_alpha=1000):
    final = []
    importance = []
    means = mu
    mean_vector = np.array(means)
    for t in range(0, num_c):
        expectation_do_x = []
        inp = copy.deepcopy(mean_vector)
        for x in np.linspace(-3, 3, num_alpha):
            inp[t] = mu[t] + x * std[t]
            input_torchvar = autograd.Variable(torch.FloatTensor(inp),
                                               requires_grad=True)
            val = get_rewards(agent=policy,
                              env=env,
                              z=input_torchvar.data,
                              max_path_length=env.spec.max_episode_length)

            expectation_do_x.append(val)
        final.append(np.array(expectation_do_x)
                     - np.mean(np.array(expectation_do_x)))
        importance.append(np.linalg.norm(np.array(expectation_do_x)))
    return final, importance
