import numpy as np

from .gym_utils import make_vec_envs


class EvogymControllerEvaluator():
    def __init__(self, env_id, structure, num_eval=1):
        self.env_id = env_id
        self.structure = structure
        self.num_eval = num_eval

    def evaluate_controller(self, key, controller, generation):
        env = make_vec_envs(self.env_id, self.structure, 0, 1)

        obs = env.reset()
        episode_rewards = []
        while len(episode_rewards) < self.num_eval:
            action = np.array(controller.activate(obs[0]))*2 - 1
            obs, _, done, infos = env.step([np.array(action)])

            if 'episode' in infos[0]:
                reward = infos[0]['episode']['r']
                episode_rewards.append(reward)

        results = {
            'fitness': np.mean(episode_rewards),
        }
        return results
