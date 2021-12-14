from src.utils import get_benchmark_by_name
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import joblib
from garage import rollout
from src.causality import plot_ace


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t] = 1
    _z, latent_info = policy.get_latent(onehot)
    _z = policy.latent_space.flatten(_z)
    return _z, latent_info["mean"], np.exp(latent_info["log_std"])


class Trainer():
    def __init__(self, args):
        self.args = args
        self._build()

    def _build(self):
        self._create_config_file()
        train_function = get_benchmark_by_name(algo_name=self.args.algo,
                                               env_name=self.args.env,)
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
        self.get_causal_attributions()
        # self._visualize_each_task()
        # self._plot_latents()

    def _build(self):
        snapshot = joblib.load(self.args.folder)
        self.env = snapshot["env"]
        self.policy = snapshot["algo"].policy
        self.baseline = snapshot["algo"]._baseline

        # Tasks and goals
        self.num_tasks = self.policy.task_space.flat_dim
        # Embedding distributions
        self.z_dists = [get_z_dist(t, self.policy) for t in range(self.num_tasks)]
        self._z = np.array([d[0] for d in self.z_dists])
        self.z_means = np.array([d[1] for d in self.z_dists])
        self.z_stds = np.array([d[2] for d in self.z_dists])
        self.num_latents = self.z_means[0].shape[0]

    def _plot_latents(self):
        fig = plt.figure(figsize=(13, 6))
        lr_grid = gridspec.GridSpec(1, 1)
        em_grid = gridspec.GridSpecFromSubplotSpec(self.num_latents, 1, subplot_spec=lr_grid[0])
        def colormap(x): return matplotlib.cm.get_cmap("Set1")(x)
        self.task_cmap = [colormap(task / (self.num_tasks-1.) * 0.5)
                          for task in range(self.num_tasks)]

        for d in range(self.num_latents):
            em_ax = plt.Subplot(fig, em_grid[d])
            em_ax.set_title("Embedding dimension %i" % d)
            em_ax.grid()
            for task in range(self.num_tasks):
                mu = self.z_means[task, d]
                sigma = self.z_stds[task, d]

                xs = np.linspace(self.latent_mins[d], self.latent_maxs[d], 100)
                ys = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                      np.exp(-0.5 * (1 / sigma * (xs - mu)) ** 2))

                em_ax.plot(xs, ys, color=self.task_cmap[task], label=f'Task: {task}')
                em_ax.fill_between(xs, np.zeros_like(xs), ys, color=self.task_cmap[task], alpha=.1)
            em_ax.legend()

            fig.add_subplot(em_ax)
        fig.tight_layout()
        fig.savefig(f"{os.path.dirname(self.args.folder)}/latent_embed.pdf")

    def _visualize_each_task(self):
        task_envs = self.env._task_envs
        for task in range(self.num_tasks):
            path = rollout(task_envs[task], self.policy,
                           max_episode_length=self.env.spec.max_episode_length,
                           animated=True)
            print(path)

    def get_causal_attributions(self):
        for task in range(self.num_tasks):
            latent_mins, latent_maxs = [], []
            for d in range(self.num_latents):
                lmin, lmax = np.inf, -np.inf
                for task in range(self.num_tasks):
                    mu = self.z_means[task, d]
                    sigma = self.z_stds[task, d]
                    lmin = min(lmin, mu-3*sigma)
                    lmax = max(lmax, mu+3*sigma)
                latent_mins.append(lmin)
                latent_maxs.append(lmax)
            self.latent_mins = latent_mins
            self.latent_maxs = latent_maxs
            X = np.zeros(shape=(1024, self.num_latents))
            for d in range(self.num_latents):
                mu = self.z_means[task, d]
                sigma = self.z_stds[task, d]
                xs = np.linspace(self.latent_mins[d], self.latent_maxs[d], 1024)
                ys = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                      np.exp(-0.5 * (1 / sigma * (xs - mu)) ** 2))
                X[:, d] = ys
            ace = plot_ace(X=X,
                           mu=self.z_means[task],
                           num_c=self.num_latents,
                           policy=self.policy,
                           env=self.env._task_envs[task],
                           baseline=self.baseline,
                           min=min(self.latent_mins),
                           max=max(self.latent_maxs))
            print("Plotted ace")
