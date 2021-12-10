from src.utils import get_benchmark_by_name
from garage import wrap_experiment, rollout
from garage.trainer import TFTrainer
# from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
import tensorflow as tf
import wandb
import os
import time
import joblib
from glob import glob


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
                      batch_size=args.batch_size, plot=True)


def _test(folder, seed=1):
    set_seed(seed)
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session():
        data = joblib.load(f'{folder}/params.pkl')
        policy = data['algo'].policy
        envs = data['env']
        # See what the trained policy can accomplish
        path = rollout(envs, policy, animated=True, deterministic=True)
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
        if self.args.train:
            self._create_config_file()
            train_function = get_benchmark_by_name(algo_name=self.args.algo,
                                                   env_name=self.args.env,)
            train_function(self.args)
        else:
            file_path = glob(self.args.snapshot_dir)[0]
            folder = os.path.join(self.args.snapshot_dir, file_path)
            _test(folder)

    def _create_config_file(self):
        if (self.args.snapshot_dir is not None):
            if not os.path.exists(self.args.snapshot_dir):
                os.makedirs(self.args.snapshot_dir)
            folder = os.path.join(self.args.snapshot_dir,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            self.args.snapshot_dir = os.path.abspath(folder)
