import numpy as np
import copy
import torch
import torch.autograd as autograd
import tensorflow as tf
from garage.experiment import deterministic


def get_action_given_latent(policy, observation, latent, network):
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
    print(flat_obs, flat_latent)
    sample, mean, log_std = network(flat_obs, flat_latent)
    sample = policy.action_space.unflatten(np.squeeze(sample, 1)[0])
    mean = policy.action_space.unflatten(np.squeeze(mean, 1)[0])
    log_std = policy.action_space.unflatten(np.squeeze(log_std, 1)[0])
    return sample, dict(mean=mean, log_std=log_std)

def get_grad_network(policy, tape):
    obs_input = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, policy.obs_dim))

    latent_input = tf.compat.v1.placeholder(
        tf.float32, shape=(None, None, policy._encoder.output_dim))
    with tf.compat.v1.variable_scope('concat_obs_latent'):
            obs_latent_input = tf.concat([obs_input, latent_input], -1)
            tape.watch(obs_latent_input[-1])
    dist, mean_var,log_std_var = super(type(policy),policy).build(
        obs_latent_input,
        # Must named 'default' to
        # compensate tf default worker
        name='given_latent').outputs
    _f_dist_obs_latent_grad = tf.compat.v1.get_default_session(
        ).make_callable([
            dist.sample(seed=deterministic.get_tf_seed_stream()), mean_var,
            log_std_var, obs_latent_input[-1]
        ],
                        feed_list=[obs_input, latent_input])

    return _f_dist_obs_latent_grad

def get_rewards(env,
                agent,
                z,
                max_path_length=np.inf,
                network=None,
                animated=False,
                speedup=1,
                always_return_paths=False):

    rewards = tf.variable(0)
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    while episode_length < (max_path_length or np.inf):
        a, agent_info = get_action_given_latent(agent, last_obs, z, network)
        a = agent_info['mean']
        es = env.step(a)
        rewards += tf.variable(es.reward)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation
    return rewards


def plot_ace(X, mu, num_c, policy, env, min, max, num_alpha=1000):
    final = []
    cov = np.cov(X, rowvar=False)
    means = mu
    network=None
    cov = np.array(cov)
    mean_vector = np.array(means)
    for t in range(0, num_c):
        expectation_do_x = []
        inp = copy.deepcopy(mean_vector)
        for x in np.linspace(min, max, num_alpha):
            inp[t] = x
            input_torchvar = inp

            last_obs, episode_infos = env.reset()
            flat_obs = policy.observation_space.flatten(last_obs)
            flat_obs = np.expand_dims([flat_obs], 1)
            flat_latent = policy.latent_space.flatten(input_torchvar.data)
            flat_latent = np.expand_dims([flat_latent], 1)
            with tf.GradientTape() as tape:
                if network is None:
                    network = get_grad_network(policy, tape)
                _,output,_, input_target=network(flat_obs, flat_latent)
                # output = get_rewards(agent=policy,
                #                      env=env,
                #                      z=input_torchvar.data,
                #                      max_path_length=env.spec.max_episode_length,
                #                      network=network)
                print(output, input_target)
                output = tf.convert_to_tensor(output, dtype=tf.float32)
                print(type(input_target), type(output))
                first_grads = tape.gradient(output, input_target)
            print(first_grads)
            o1 = output.data.cpu()
            val = o1.numpy()[0]
            grad_mask_gradient = torch.zeros(1)
            grad_mask_gradient[0] = 1.0
            # first_grads = torch.autograd.grad(output.cpu(), input_torchvar.cpu(
            # ), grad_outputs=grad_mask_gradient, retain_graph=True,
            #     create_graph=True)
            # input_torchvar = tf.Variable(inp)
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
