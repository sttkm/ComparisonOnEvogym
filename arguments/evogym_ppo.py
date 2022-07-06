import argparse

from evogym import BASELINE_ENV_NAMES

def get_args():
    parser = argparse.ArgumentParser(
        description='Evogym PPO experiment'
    )

    parser.add_argument(
        '-n', '--name',
        default='', type=str,
        help='experiment name (default: "{task}_{robot}")'
    )
    parser.add_argument(
        '-t', '--task',
        default='Walker-v0', type=str,
        help='evogym environment id (default: Walker-v0)'
    )
    parser.add_argument(
        '-r', '--robot',
        default='default', type=str,
        help='robot structure name (default: default, built on "robot_files/", if "default", load default robot for the task)'
    )

    parser.add_argument(
        '-p', '--num-processes',
        default=4, type=int,
        help='how many training CPU processes to use (default: 4)'
    )
    parser.add_argument(
        '-s', '--steps',
        default=128, type=int,
        help='num steps to use in PPO (default: 128)'
    )
    parser.add_argument(
        '-b','--num-mini-batch',
        default=4, type=int,
        help='number of batches for ppo (default: 4)'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=4, type=int,
        help='number of ppo epochs (default: 4)'
    )
    parser.add_argument(
        '-i', '--ppo-iters',
        default=100, type=int,
        help='learning iterations of PPO (default: 100)'
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        default=2.5e-4, type=float,
        help='learning rate (default: 2.5e-4)'
    )
    parser.add_argument(
        '--gamma',
        default=0.99, type=float,
        help='discount factor for rewards (default: 0.99)'
    )
    parser.add_argument(
        '-c', '--clip-range',
        default=0.1, type=float,
        help='ppo clip parameter (default: 0.1)'
    )
    parser.add_argument(
        '-pb', '--probabilistic',
        action='store_true', default=False,
        help='robot act probabilistic (default: False)'
    )
    parser.add_argument(
        '--view',
        action='store_true', default=False,
        help='open simulation window of best robot (default: False)'
    )
    args = parser.parse_args()

    if args.name=='':
        args.name = f'{args.task}_{args.robot}'

    assert args.task in BASELINE_ENV_NAMES,\
        f'argumented task id "{args.task}" is not prepared, pick from ['+', '.join(BASELINE_ENV_NAMES)+'].'

    return args


def get_gif_args():
    parser = argparse.ArgumentParser(
        description='make robot gifs'
    )

    parser.add_argument(
        'name',
        type=str,
        help='name of experiment for making figures'
    )
    parser.add_argument(
        '-s', '--specified',
        type=int,
        help='input iter, make figure for the only specified controller (usage: "-s {iter}")'
    )

    parser.add_argument(
        '-r', '--resolution-ratio',
        default=0.2, type=float,
        help='gif resolution ratio (default: 0.2 -> 256:144)'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=1, type=int,
        help='number of parallel making processes (default: 1)'
    )
    parser.add_argument(
        '--not-overwrite',
        action='store_true', default=False,
        help='skip process if already figure exists (default: False)'
    )
    parser.add_argument(
        '--no-multi',
        action='store_true', default=False,
        help='do without using multiprocessing. if error occur, try this option. (default: False)'
    )

    args = parser.parse_args()

    assert args.name is not None, 'argumented error: input "{experiment name}"'

    return args
