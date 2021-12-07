from src.utils import get_benchmark_by_name
from garage import rollout, wrap_experiment
from garage.trainer import TFTrainer
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
import tensorflow as tf
import wandb
import os
import time


@wrap_experiment()
def _train(ctxt, seed=1):
    args = config
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        benchmark = get_benchmark_by_name(algo_name=args.algo,
                                          algo_args=args.algo_args,
                                          env_name=args.env,
                                          env_args=args.env_args)
        trainer.setup(benchmark.algo, benchmark.env)
        trainer.train(n_epochs=args.epochs,
                      batch_size=args.batch_size_per_task,
                      plot=args.plot)


def _test(snapshot_dir, seed=1):
    set_seed(seed)
    snapshotter = Snapshotter()
    with tf.compat.v1.Session():
        data = snapshotter.load(snapshot_dir)
        policy = data['algo'].policy
        env = data['env']
        # See what the trained policy can accomplish
        path = rollout(env, policy, animated=True)
        print(path)


class Trainer():
    def __init__(self, args):
        global config
        config = args
        self.args = args
        wandb.init(project='Task_Structure', entity='td_ml', config=args, settings=wandb.Settings(start_method='thread'),
                   name=args.exp_name, reinit=False)
        self._build()

    def _build(self):
        self._create_config_file()
        if self.args.train:
            _train({'log_dir': self.args.snapshot_dir, 'use_existing_dir': True}, seed=self.args.seed)
        else:
            _test(snapshot_dir=self.args.snapshot_dir)

    def _create_config_file(self):
        if (self.args.snapshot_dir is not None):
            if not os.path.exists(self.args.snapshot_dir):
                os.makedirs(self.args.snapshot_dir)
            folder = os.path.join(self.args.snapshot_dir,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            self.args.snapshot_dir = os.path.abspath(folder)
