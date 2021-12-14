from src.launchers import train_te_ppo_pointenv, train_ate_ppo_pointenv
from src.launchers import train_te_ppo_mt10, train_ate_ppo_mt10
import numpy as np


def stack_tensor_dict_list(tensor_dict_list):
    """Stack a list of dictionaries of {tensors or dictionary of tensors}.
    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.
    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = stack_tensor_dict_list(dict_list)
        else:
            v = np.array(dict_list)

        ret[k] = v

    return ret


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
