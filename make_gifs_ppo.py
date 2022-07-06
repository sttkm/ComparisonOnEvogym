import os
from glob import glob

import multiprocessing as mp


import evogym.envs

from experiment_utils import load_experiment, load_robot

from utils.gif_maker import EvogymControllerDrawerPPO, pool_init_func


from arguments.evogym_ppo import get_gif_args


def main():

    args = get_gif_args()

    expt_path = os.path.join('out', 'evogym_ppo', args.name)
    expt_args = load_experiment(expt_path)


    structure = load_robot(expt_args['robot'], task=expt_args['task'])


    controller_path = os.path.join(expt_path, 'controller')
    if args.specified is not None:
        controller_files = [os.path.join(controller_path, f'{args.specified}.zip')]
    else:
        controller_files = glob(os.path.join(controller_path, '*.zip'))


    draw_kwargs = {
        'resolution': (1280*args.resolution_ratio, 720*args.resolution_ratio),
        'deterministic': not expt_args['probabilistic']
    }
    drawer = EvogymControllerDrawerPPO(
        save_path=expt_path,
        env_id=expt_args['task'],
        structure=structure,
        overwrite=not args.not_overwrite,
        **draw_kwargs)

    draw_function = drawer.draw


    if not args.no_multi and args.specified is None:

        lock = mp.Lock()
        pool = mp.Pool(args.num_cores, initializer=pool_init_func, initargs=(lock,))
        jobs = []

        for controller_file in controller_files:
            iter = int(os.path.splitext(os.path.basename(controller_file))[0])
            jobs.append(pool.apply_async(draw_function, args=(iter, controller_file)))

        for job in jobs:
            job.get(timeout=None)


    else:

        lock = mp.Lock()
        lock = pool_init_func(lock)

        for controller_file in controller_files:
            iter = int(os.path.splitext(os.path.basename(controller_file))[0])
            draw_function(iter, controller_file)

if __name__=='__main__':
    main()
