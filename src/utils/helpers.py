import gym
import torch
import torch.nn.functional as F
from functools import reduce
from operator import mul

from src.maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from src.ciayn.model import Embedding_Network, Inference_Network


def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = get_input_size(env)
    nonlinearity = getattr(torch, nonlinearity)

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size,
                                 output_size,
                                 hidden_sizes=tuple(hidden_sizes),
                                 nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size,
                                      output_size,
                                      hidden_sizes=tuple(hidden_sizes),
                                      nonlinearity=nonlinearity)
    return policy


def get_encoder(input_size=10, output_size=64, hidden_sizes=(64), nonlinearity=F.elu):
    return Embedding_Network(input_size, output_size, hidden_sizes, nonlinearity)


def get_inference(input_size=10, output_size=64, hidden_sizes=(200, 200), nonlinearity=F.elu):
    return Inference_Network(input_size, output_size, hidden_sizes, nonlinearity)


def get_input_size(env):
    return reduce(mul, env.observation_space.shape, 1)
