import os
import csv

import multiprocessing as mp


import evogym.envs

import neat_
from experiment_utils import load_experiment, load_robot

from utils.gif_maker import EvogymControllerDrawerNEAT, pool_init_func


from arguments.evogym_neat import get_gif_args


def main():

    args = get_gif_args()

    expt_path = os.path.join('out', 'evogym_neat', args.name)
    expt_args = load_experiment(expt_path)


    structure = load_robot(expt_args['robot'], task=expt_args['task'])


    decode_function = neat_.FeedForwardNetwork.create


    config_file = os.path.join(expt_path, 'evogym_neat.cfg')
    config = neat_.make_config(config_file)


    genome_path = os.path.join(expt_path, 'genome')
    genome_ids = {}
    if args.specified is not None:
        genome_ids = {
            'specified': [args.specified]
        }
    else:
        files = {
            'reward': 'history_reward.csv',
        }
        for metric,file in files.items():

            history_file = os.path.join(expt_path, file)
            with open(history_file, 'r') as f:
                reader = csv.reader(f)
                histories = list(reader)[1:]
                ids = sorted(list(set([hist[1] for hist in histories])))
                genome_ids[metric] = ids


    draw_kwargs = {
        'resolution': (1280*args.resolution_ratio, 720*args.resolution_ratio),
    }
    drawer = EvogymControllerDrawerNEAT(
        save_path=expt_path,
        env_id=expt_args['task'],
        structure=structure,
        genome_config=config.genome_config,
        decode_function=decode_function,
        overwrite=not args.not_overwrite,
        **draw_kwargs)

    draw_function = drawer.draw


    if not args.no_multi and args.specified is None:

        lock = mp.Lock()
        pool = mp.Pool(args.num_cores, initializer=pool_init_func, initargs=(lock,))
        jobs = []

        for metric,ids in genome_ids.items():
            for key in ids:
                genome_file = os.path.join(genome_path, f'{key}.pickle')
                jobs.append(pool.apply_async(draw_function, args=(key, genome_file), kwds={'directory': metric}))

        for job in jobs:
            job.get(timeout=None)


    else:

        lock = mp.Lock()
        lock = pool_init_func(lock)

        for metric,ids in genome_ids.items():
            for key in ids:
                genome_file = os.path.join(genome_path, f'{key}.pickle')
                draw_function(key, genome_file, directory=metric)

if __name__=='__main__':
    main()
