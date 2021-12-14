import argparse
from src.trainer import Trainer, Tester
from glob import glob
import logging
import wandb


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
                        choices=['point_mass', 'mt10'],
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
    parser.add_argument('--policy_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of policy optimizer (default: 1e-3)')
    parser.add_argument('--encoder_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of encoder optimizer (default: 1e-3)')
    parser.add_argument('--inference_optimizer_lr', type=float, default=1e-3,
                        help='Learning rate of inference optimizer (default: 1e-3)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')

    args, unknownargs = parser.parse_known_args()
    args.snapshot_dir = f"{args.snapshot_dir}{args.algo}/{args.env}/"
    return args


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project='Task_Structure', entity='td_ml', config=args, settings=wandb.Settings(start_method='thread'),
               name=args.exp_name, reinit=False)
    logging.info(args)
    if args.train:
        Trainer(args)
    else:
        for run, config_file in enumerate(glob(f'{args.snapshot_dir}/*/params.pkl')):
            args.folder = config_file
            logging.info(config_file)
            Tester(args)
