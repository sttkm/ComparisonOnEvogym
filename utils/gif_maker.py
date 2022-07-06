import os
import pickle
import numpy as np

import imageio
from pygifsicle import gifsicle

from .gym_utils import make_vec_envs

from stable_baselines3 import PPO



def pool_init_func(lock_):
    global lock
    lock = lock_


def make_gif(filename, env, viewer, controller, controller_type, resolution=(256,144), deterministic=True):
    assert controller_type in ['NEAT', 'PPO']

    viewer.set_resolution(resolution)

    done = False
    obs = env.reset()
    imgs = []
    while not done:

        img = viewer.render(mode='img')
        imgs.append(img)

        if controller_type=='NEAT':
            action = [np.array(controller.activate(obs[0]))*2 - 1]
        elif controller_type=='PPO':
            action, _ = controller.predict(obs, deterministic=deterministic)
        else:
            return
        obs, _, done, infos = env.step(action)


    imageio.mimsave(filename, imgs, duration=(1/50.0))

    with lock:
        gifsicle(sources=filename,
                 destination=filename,
                 optimize=False,
                 colors=64,
                 options=["--optimize=3","--no-warnings"])

    return

class EvogymControllerDrawerNEAT():
    def __init__(self, save_path, env_id, structure, genome_config, decode_function, overwrite=True, **draw_kwargs):
        self.save_path = os.path.join(save_path, 'gif')
        self.env_id = env_id
        self.structure = structure
        self.genome_config = genome_config
        self.decode_function = decode_function
        self.overwrite = overwrite
        self.draw_kwargs = draw_kwargs

        os.makedirs(self.save_path, exist_ok=True)

    def draw(self, key, genome_file, directory=''):
        save_dir = os.path.join(self.save_path, directory)
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'{key}.gif')
        if not self.overwrite and os.path.exists(filename):
            return

        env = make_vec_envs(self.env_id, self.structure, 0, 1, allow_early_resets=False)
        viewer = env.get_attr("default_viewer", indices=None)[0]

        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)
        controller = self.decode_function(genome, self.genome_config)

        make_gif(filename, env, viewer, controller, 'NEAT', **self.draw_kwargs)

        env.close()
        print(f'genome {key} ... done')
        return


class EvogymControllerDrawerPPO():
    def __init__(self, save_path, env_id, structure, overwrite=True, **draw_kwargs):

        self.save_path = os.path.join(save_path, 'gif')
        self.env_id = env_id
        self.structure = structure
        self.overwrite = overwrite
        self.draw_kwargs = draw_kwargs

        os.makedirs(self.save_path, exist_ok=True)

    def draw(self, iter, ppo_file, directory=''):
        save_dir = os.path.join(self.save_path, directory)
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'{iter}.gif')
        if not self.overwrite and os.path.exists(filename):
            return

        env = make_vec_envs(self.env_id, self.structure, 0, 1, allow_early_resets=False, vecnormalize=True)
        viewer = env.get_attr("default_viewer", indices=None)[0]

        controller = PPO.load(ppo_file)
        env.obs_rms = controller.env.obs_rms

        make_gif(filename, env, viewer, controller, 'PPO', **self.draw_kwargs)

        env.close()
        print(f'iter {iter} ... done')
        return
