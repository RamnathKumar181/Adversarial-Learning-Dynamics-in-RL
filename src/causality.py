import numpy as np
import copy
from src.utils import stack_tensor_dict_list
import torch
import torch.autograd as autograd


def get_action_given_latent(policy, observation, latent):
    """Sample an action given observation and latent.
    Args:
        observation (np.ndarray): Observation from the environment,
            with shape :math:`(O, )`. O is the dimension of observation.
        latent (np.ndarray): Latent, with shape :math:`(Z, )`. Z is the
            dimension of the latent embedding.
    Returns:
        np.ndarray: Action sampled from the policy,
            with shape :math:`(A, )`. A is the dimension of action.
        dict: Action distribution information, with keys:
            - mean (numpy.ndarray): Mean of the distribution,
                with shape :math:`(A, )`. A is the dimension of action.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution, with shape :math:`(A, )`. A is the dimension
                of action.
    """
    flat_obs = policy.observation_space.flatten(observation)
    flat_obs = np.expand_dims([flat_obs], 1)
    flat_latent = policy.latent_space.flatten(latent)
    flat_latent = np.expand_dims([flat_latent], 1)
    sample, mean, log_std = policy._f_dist_obs_latent(flat_obs, flat_latent)
    sample = policy.action_space.unflatten(np.squeeze(sample, 1)[0])
    mean = policy.action_space.unflatten(np.squeeze(mean, 1)[0])
    log_std = policy.action_space.unflatten(np.squeeze(log_std, 1)[0])
    return sample, dict(mean=mean, log_std=log_std)


def get_path(env,
             agent,
             z,
             max_path_length=np.inf,
             animated=False,
             speedup=1,
             always_return_paths=False):

    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize()
    while episode_length < (max_path_length or np.inf):
        a, agent_info = agent.get_action_given_latent(last_obs, z.data)
        a = agent_info['mean']
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )


def plot_ace(X, mu, num_c, policy, env, baseline, min, max, num_alpha=1000):
    final = []
    cov = np.cov(X, rowvar=False)
    means = mu
    cov = np.array(cov)
    mean_vector = np.array(means)
    for t in range(0, num_c):
        expectation_do_x = []
        inp = copy.deepcopy(mean_vector)
        for x in np.linspace(min, max, num_alpha):
            inp[t] = x
            input_torchvar = autograd.Variable(torch.FloatTensor(inp),
                                               requires_grad=True)
            paths = get_path(agent=policy,
                             env=env,
                             z=input_torchvar,
                             max_path_length=env.spec.max_episode_length)
            featmat = np.concatenate([baseline._features(path) for path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            print(output)
            o1 = output.data.cpu()
            val = o1.numpy()[0]
            grad_mask_gradient = torch.zeros(1)
            grad_mask_gradient[0] = 1.0
            first_grads = torch.autograd.grad(output.cpu(), input_torchvar.cpu(
            ), grad_outputs=grad_mask_gradient, retain_graph=True,
                create_graph=True)
            # input_torchvar = tf.Variable(inp)
            # with tf.GradientTape() as tape:
            #     output = get_rewards(agent=policy,
            #                          env=env,
            #                          z=input_torchvar,
            #                          max_path_length=env.spec.max_episode_length)
            #     first_grads = tape.gradient(output, input_torchvar)
            for dimension in range(0, num_c):  # Tr(Hessian*Covariance)
                if dimension == t:
                    continue
                temp_cov = copy.deepcopy(cov)
                temp_cov[dimension][t] = 0.0
                grad_mask_hessian = torch.zeros(num_c)
                grad_mask_hessian[dimension] = 1.0
                # calculating the hessian
                hessian = torch.autograd.grad(
                    first_grads, input_torchvar,
                    grad_outputs=grad_mask_hessian,
                    retain_graph=True, create_graph=False)
                # adding second term in interventional expectation
                val += np.sum(0.5*hessian[0].data.numpy()*temp_cov[dimension])
            # append interventional expectation for given interventional value
            expectation_do_x.append(val)
        final.append(np.array(expectation_do_x) -
                     np.mean(np.array(expectation_do_x)))
    return final
