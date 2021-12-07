import argparse
from src.trainer import Trainer


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('Adversarial Task Learning')
    # General
    parser.add_argument('--algo', type=str,
                        choices=['te_ppo', 'ate_ppo'],
                        default='te_ppo',
                        help='Algorithm to be used (default: te_ppo).')
    parser.add_argument('--env', type=str,
                        choices=['point_mass'],
                        default='point_mass',
                        help='Environment to be used (default: point_mass).')
    parser.add_argument('--train', action='store_true',
                        help='Train the model (default: False).')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs to run for (default: 600)')
    parser.add_argument('--batch_size_per_task', type=int, default=1024,
                        help='Batch size per task (default: 1024)')
    parser.add_argument('--snapshot_dir', type=str, default='config/',
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name'
                        '(default: None).')

    # Environment Args
    env_args = argparse.ArgumentParser('Environment args')
    env_args.add_argument('--n', type=int, default=4,
                          help='Number of points in point mass env. (default: 4).')

    # ALgorithm Args
    algo_args = argparse.ArgumentParser('Algorithm args')
    algo_args.add_argument('--embedding_init_std', type=float, default=0.1,
                           help='Embedding init std. (default: 0.1).')
    algo_args.add_argument('--embedding_max_std', type=float, default=0.2,
                           help='Embedding max std. (default: 0.2).')
    algo_args.add_argument('--embedding_min_std', type=float, default=1e-6,
                           help='Embedding min std. (default: 1e-6).')
    algo_args.add_argument('--policy_init_std', type=float, default=0.1,
                           help='Policy init std. (default: 0.1).')
    algo_args.add_argument('--policy_min_std', type=float, default=None,
                           help='Policy min std. (default: None).')
    algo_args.add_argument('--policy_max_std', type=float, default=2.0,
                           help='Policy max std. (default: 2.0).')
    algo_args.add_argument('--policy_ent_coeff', type=float, default=1e-3,
                           help='Policy entropy coefficient. (default: 1e-3).')
    algo_args.add_argument('--encoder_ent_coeff', type=float, default=1e-3,
                           help='Encoder entropy coefficient. (default: 1e-3).')
    algo_args.add_argument('--inference_ce_coeff', type=float, default=5e-2,
                           help='Inference cross-entropy coefficient. (default: 5e-2).')
    algo_args.add_argument('--inference_window', type=int, default=6,
                           help='Inference window size. (default: 6).')
    algo_args.add_argument('--latent_length', type=int, default=2,
                           help='Latent length. (default: 2).')
    algo_args.add_argument('--steps_per_epoch', type=int, default=10,
                           help='Steps per epoch. (default: 10).')
    algo_args.add_argument('--sampler_batch_size', type=int, default=500,
                           help='Sampler batch size. (default: 500).')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu).')
    misc.add_argument('--plot', action='store_true',
                      help='Use snapshotter (default: false, do not snapshot).')

    args, unknownargs = parser.parse_known_args()
    args.env_args, unknownargs = env_args.parse_known_args(unknownargs)
    args.algo_args, _ = algo_args.parse_known_args(unknownargs)
    args.snapshot_dir = f"{args.snapshot_dir}{args.algo}/{args.env}/"
    args.algo_args.epochs = args.epochs
    return args


if __name__ == '__main__':
    args = parse_args()
    Trainer(args)
