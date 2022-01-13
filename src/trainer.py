from src.utils import get_benchmark_by_name, seed_everything
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import joblib
import cloudpickle
from garage import rollout
from src.causality import plot_ace
import matplotlib as mpl
from garage.experiment.deterministic import set_seed
import scipy.fftpack


def rollout_given_z(env,
                    agent,
                    z,
                    max_path_length=np.inf,
                    animated=False,
                    speedup=1):
    o, episode_infos = env.reset()
    agent.reset()

    if animated:
        env.visualize()
    path_length = 0
    observations = []
    while path_length < max_path_length:
        a, agent_info = agent.get_action_given_latent(
            o, z)
        es = env.step(a)
        observations.append(o)
        path_length += 1
        if es.last:
            break
        o = es.observation
    return np.array(observations)


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t-1] = 1
    _, latent_info = policy.get_latent(onehot)
    return latent_info["mean"], np.exp(latent_info["log_std"])


class Trainer():
    def __init__(self, args):
        self.args = args
        self._build()

    def _build(self):
        seed_everything()
        self._create_config_file()
        train_function = get_benchmark_by_name(algo_name=self.args.algo,
                                               env_name=self.args.env)
        train_function(self.args)

    def _create_config_file(self):
        if (self.args.snapshot_dir is not None):
            if not os.path.exists(self.args.snapshot_dir):
                os.makedirs(self.args.snapshot_dir)
            folder = os.path.join(self.args.snapshot_dir,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            self.args.snapshot_dir = os.path.abspath(folder)


class Tester():
    def __init__(self, args):
        self.args = args
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.InteractiveSession()
        self._build()
        if args.vis:
            self._visualize_each_task()
        if args.causal:
            self.get_causal_attributions()

    def _build(self):
        with open(self.args.folder, 'rb') as file:
            snapshot = cloudpickle.load(file)
        self.env = snapshot["env"]
        self.policy = snapshot["algo"].policy

        # Tasks and goals
        self.num_tasks = self.policy.task_space.flat_dim
        # Embedding distributions
        self.z_dists = [get_z_dist(t, self.policy)
                        for t in range(self.num_tasks)]
        self.z_means = np.array([d[0] for d in self.z_dists])
        self.z_stds = np.array([d[1] for d in self.z_dists])
        self.num_latents = self.z_means[0].shape[0]

        self.colormap = mpl.cm.Dark2.colors

    def _visualize_each_task(self):
        task_envs = self.env._task_envs
        fig = plt.figure(figsize=(7, 7))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        for task in range(self.num_tasks):
            print(f"{task}: {task_envs[task]._goal}")
            plt.scatter(task_envs[task]._goal[0], task_envs[task]._goal[1],
                        s=5000, color=self.colormap[task], alpha=0.3)
            for i in range(5):
                set_seed(i)
                path = rollout_given_z(task_envs[task], self.policy,
                                       self.z_means[task],
                                       max_path_length=200,
                                       animated=True)
                plt.plot(path[:, 0], path[:, 1],
                         alpha=0.7, color=self.colormap[task])
        fig.tight_layout()
        fig.savefig(f"{os.path.dirname(self.args.folder)}/rollout.pdf")

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def get_causal_attributions(self):
        for task in range(self.num_tasks):
            ace_total = []
            imp_total = []
            for run in range(5):
                ace, imp = plot_ace(mu=self.z_means[task],
                                    std=self.z_stds[task],
                                    num_c=self.num_latents,
                                    policy=self.policy,
                                    env=self.env._task_envs[task])
                imp = (imp - min(imp))/(max(imp)-min(imp))
                ace_total.append(ace)
                imp_total.append(imp)
            mean_ace = np.mean(ace_total, axis=0)
            std_ace = np.std(ace_total, axis=0)

            mean_imp = np.mean(imp_total, axis=0)
            std_imp = np.std(imp_total, axis=0)

            fig = plt.figure(figsize=(10, 10))
            plt.xlabel('Intervention Value (alpha)', fontsize=26)
            plt.ylabel('Causal Attributions (ACE)', fontsize=26)
            for t in range(0, self.num_latents):
                mace = self.smooth(mean_ace[t], 50)
                plt.plot(np.linspace(self.z_means[task][t]-3, self.z_means[task][t] + 3, 1000),
                         mace,
                         label=r'$z_{}$'.format(t+1),
                         color=self.colormap[t])
                plt.fill_between(np.linspace(self.z_means[task][t]-3, self.z_means[task][t] + 3, 1000),
                                 self.smooth(mean_ace[t]+std_ace[t], 50), mace,
                                 color=self.colormap[t], alpha=0.3)
                plt.fill_between(np.linspace(self.z_means[task][t]-3, self.z_means[task][t] + 3, 1000),
                                 self.smooth(mean_ace[t]-std_ace[t], 50), mace,
                                 color=self.colormap[t], alpha=0.3)
            plt.legend(fontsize=22)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            fig.tight_layout()
            fig.savefig(f"{os.path.dirname(self.args.folder)}/ace_{task}.pdf")
            print(
                f"Plotted ace for task: {task} with goal: {self.env._task_envs[task]._goal} and latent: {self.z_means[task]}")
            print(f"Importance for task {task}/mean: {mean_imp}")
            print(f"Importance for task {task}/std: {std_imp}")
