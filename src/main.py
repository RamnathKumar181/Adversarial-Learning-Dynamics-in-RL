import torch
import wandb
import argparse
import multiprocessing as mp
import json
import os
import yaml
import gc
from src.utils.torch_utils import seed_everything
from src.maml_rl.trainer import MAMLTrainer, MAMLTester


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('Task_Structure')
    # General
    parser.add_argument('--config', type=str, required=True,
                        help='path to the configuration file.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--policy', type=str, required=False,
                        help='path to the policy checkpoint')
    parser.add_argument('--model', type=str,
                        choices=['maml', 'protonet', 'reptile',
                                 'matching_networks', 'cnaps', 'metaoptnet'],
                        default='maml',
                        help='Name of the model to be used (default: MAML).')

    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name'
                        '(default: None).')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
                      help='name of the output folder')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                      '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
                      'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['num_workers'] = args.num_workers
    config['device'] = args.device
    config['seed'] = args.seed
    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        config['policy_filename'] = os.path.join(args.output_folder, 'policy.th')
        config['config_filename'] = os.path.join(args.output_folder, 'config.json')

        with open(config['config_filename'], 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    if args.train:
        wandb.init(project='Task_Structure', config=config, name=args.exp_name,
                   settings=wandb.Settings(start_method='thread'), reinit=False)
        if args.model == 'maml':
            """
            MAML Trainer
            """
            gc.collect()
            seed_everything(0)
            maml_trainer = MAMLTrainer(config)

    wandb.init(project='Task_Structure', config=config, name=args.exp_name,
               settings=wandb.Settings(start_method='thread'), reinit=False)
    if args.model == 'maml':
        """
        MAML Trainer
        """
        gc.collect()
        seed_everything(0)
        maml_trainer = MAMLTester(config)
