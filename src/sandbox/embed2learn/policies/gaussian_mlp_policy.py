import itertools

import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.distributions import DiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.spaces import Box
from garage.tf.policies import StochasticPolicy
from sandbox.embed2learn.tf.network_utils import mlp
from sandbox.embed2learn.tf.network_utils import parameter


class GaussianMLPPolicy(StochasticPolicy, Parameterized, Serializable):
    def __init__(self,
                 env_spec,
                 name="GaussianMLPPolicy",
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 mean_network=None,
                 std_network=None,
                 std_parameterization='exp'):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden
          layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers
          for std
        :param min_std: whether to make sure that the std is at least some
          threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There
          are a few options:
            - exp: the logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        assert isinstance(env_spec.action_space, Box)
        StochasticPolicy.__init__(self, env_spec)
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        if mean_network or std_network:
            raise NotImplementedError

        self.name = name
        self._variable_scope = tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE)
        self._name_scope = tf.name_scope(self.name)

        # TODO: eliminate
        self._dist = DiagonalGaussian(self.action_space.flat_dim)

        # Network parameters
        self._hidden_sizes = hidden_sizes
        self._learn_std = learn_std
        self._init_std = init_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._mean_network = mean_network
        self._std_network = std_network
        self._std_parameterization = std_parameterization

        # Tranform std arguments to parameterized space
        self._init_std_param = None
        self._min_std_param = None
        self._max_std_param = None
        if self._std_parameterization == 'exp':
            self._init_std_param = np.log(init_std)
            if min_std:
                self._min_std_param = np.log(min_std)
            if max_std:
                self._max_std_param = np.log(max_std)
        elif self._std_parameterization == 'softplus':
            self._init_std_param = np.log(np.exp(init_std) - 1)
            if min_std:
                self._min_std_param = np.log(np.exp(min_std) - 1)
            if max_std:
                self._max_std_param = np.log(np.exp(max_std) - 1)
        else:
            raise NotImplementedError

        # Build default graph
        with self._name_scope:
            # inputs
            self._obs_input = self.observation_space.new_tensor_variable(
                name="obs_input", extra_dims=1)

            with tf.name_scope(
                    "default", values=[self._obs_input]):
                action_var, mean_var, std_param_var, dist = self._build_graph(
                    self._obs_input)

                # outputs
                self._action = tf.identity(action_var, name="action")
                self._action_mean = tf.identity(mean_var, name="action_mean")
                self._action_std_param = tf.identity(std_param_var,
                                                     "action_std_param")
                self._action_distribution = dist

            # compiled functions
            with tf.variable_scope("f_dist"):
                self.f_dist = tensor_utils.compile_function(
                    inputs=[self._obs_input],
                    outputs=[
                        self._action,
                        self._action_mean,
                        self._action_std_param
                    ],
                )

    @property
    def inputs(self):
        return self._obs_input

    @property
    def outputs(self):
        return self._action, self._action_std_param, self._dist

    def _build_graph(self, from_obs_input):
        action_dim = self.action_space.flat_dim
        small = 1e-5

        with self._variable_scope:

            with tf.variable_scope("dist_params"):
                if self._std_share_network:
                    # mean and std networks share an MLP
                    b = np.concatenate(
                        [
                            np.zeros(action_dim),
                            np.full(action_dim, self._init_std_param)
                        ],
                        axis=0)
                    b = tf.constant_initializer(b)
                    mean_std_network = mlp(
                        with_input=from_obs_input,
                        output_dim=action_dim * 2,
                        hidden_sizes=self._hidden_sizes,
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        # hidden_w_init=tf.orthogonal_initializer(1.0),
                        # output_w_init=tf.orthogonal_initializer(0.1),
                        output_b_init=b,
                        name="mean_std_network")
                    with tf.variable_scope("mean_network"):
                        mean_network = mean_std_network[..., :action_dim]
                    with tf.variable_scope("std_network"):
                        std_network = mean_std_network[..., action_dim:]

                else:
                    # separate MLPs for mean and std networks
                    # mean network
                    mean_network = mlp(
                        with_input=from_obs_input,
                        output_dim=action_dim,
                        hidden_sizes=self._hidden_sizes,
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        name="mean_network")

                    # std network
                    if self._adaptive_std:
                        b = tf.constant_initializer(self._init_std_param)
                        std_network = mlp(
                            with_input=from_obs_input,
                            output_dim=action_dim,
                            hidden_sizes=self._std_hidden_sizes,
                            hidden_nonlinearity=self._std_hidden_nonlinearity,
                            output_nonlinearity=self._output_nonlinearity,
                            output_b_init=b,
                            name="std_network")
                    else:
                        p = tf.constant_initializer(self._init_std_param)
                        std_network = parameter(
                            with_input=from_obs_input,
                            length=action_dim,
                            initializer=p,
                            trainable=self._learn_std,
                            name="std_network")

                mean_var = mean_network
                std_param_var = std_network

                with tf.variable_scope("std_limits"):
                    if self._min_std_param:
                        std_param_var = tf.maximum(std_param_var,
                                                   self._min_std_param)
                    if self._max_std_param:
                        std_param_var = tf.minimum(std_param_var,
                                                   self._max_std_param)

            with tf.variable_scope("std_parameterization"):
                # build std_var with std parameterization
                if self._std_parameterization == "exp":
                    std_var = tf.exp(std_param_var)
                elif self._std_parameterization == "softplus":
                    std_var = tf.log(1. + tf.exp(std_param_var))
                else:
                    raise NotImplementedError

            dist = tf.contrib.distributions.MultivariateNormalDiag(
                mean_var, std_var)

            action_var = dist.sample(seed=ext.get_seed())

            return action_var, mean_var, std_param_var, dist

    @property
    def vectorized(self):
        return True

    @overrides
    def get_params_internal(self, **tags):
        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self.name)]
        else:
            params = [v for v in tf.global_variables(scope=self.name)]

        return params

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        with tf.name_scope(name, "dist_info_sym", [obs_var, state_info_vars]):
            _, mean_var, log_std_var, _ = self._build_graph(obs_var)

            return dict(mean=mean_var, log_std=log_std_var)

    def entropy_sym(self, obs_var, name=None):
        with tf.name_scope(name, "entropy_sym", [obs_var]):
            _, _, _, dist = self._build_graph(obs_var)
            return dist.entropy()

    @overrides
    def get_action(self, observation):
        """
        :param observation:
        :return: action, dict
        """
        flat_obs = self.observation_space.flatten(observation)
        action, mean, log_std = [x[0] for x in self.f_dist([flat_obs])]
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions, means, log_stds = self.f_dist(flat_obs)
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self,
                               obs_var,
                               action_var,
                               old_dist_info_vars,
                               name="get_reparam_action_sym"):
        """
        Given observations, old actions, and distribution of old actions,
        return a symbolically reparameterized representation of the actions in
        terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        with tensor_utils.enclosing_scope(self.name, name):
            new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
            new_mean_var, new_log_std_var = new_dist_info_vars[
                "mean"], new_dist_info_vars["log_std"]
            old_mean_var, old_log_std_var = old_dist_info_vars[
                "mean"], old_dist_info_vars["log_std"]
            epsilon_var = (action_var - old_mean_var) / (
                tf.exp(old_log_std_var) + 1e-8)
            new_action_var = new_mean_var + epsilon_var * tf.exp(
                new_log_std_var)
            return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
