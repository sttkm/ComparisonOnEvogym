import os
import csv
import time
import numpy as np

from stable_baselines3 import PPO

from .gym_utils import make_vec_envs

def evaluate(model, envs, num_eval=1, deterministic=True):

    obs = envs.reset()
    episode_rewards = []
    while len(episode_rewards) < num_eval:
        action, _state = model.predict(obs,deterministic=deterministic)
        obs, _, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info:
                episode_rewards.append(info['episode']['r'])
    return np.mean(episode_rewards)


def run_ppo(env_id, structure, train_iters, config, save_path, history_file, deterministic=True):

    train_envs = make_vec_envs(env_id, structure, config.seed, config.num_processes, vecnormalize=True)
    train_envs.reset()

    eval_envs = make_vec_envs(env_id, structure, config.seed, config.eval_processes, vecnormalize=True)

    model = PPO(
        "MlpPolicy",
        train_envs,
        n_steps = config.steps * config.num_processes,
        batch_size = config.steps // config.num_mini_batch * config.num_processes,
        n_epochs = config.epochs,
        learning_rate = config.learning_rate,
        verbose = 0,
        gamma = config.gamma,
        clip_range = config.clip_range,
        ent_coef = config.ent_coef,
        policy_kwargs = config.policy_kwargs)


    interval = time.time()
    model.save(os.path.join(save_path, '0'), include=['env'])

    history_header = ['iteration', 'reward']
    items = {
        'iteration': 0,
        'reward': 0
    }
    with open(history_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=history_header)
        writer.writeheader()
        writer.writerow(items)


    steps_by_iter = config.learning_steps * config.steps * config.num_processes
    max_reward = float('-inf')

    for i in range(train_iters):

        model.learn(total_timesteps=steps_by_iter)

        eval_envs.obs_rms = train_envs.obs_rms
        reward = evaluate(model, eval_envs, num_eval=config.eval_processes, deterministic=deterministic)

        now = time.time()
        print(f'iteration: {i+1}  elapsed times: {now-interval: .3f}')
        interval = now

        model.save(os.path.join(save_path, str(i+1)), include=['env'])

        items = {
            'iteration': i+1,
            'reward': reward
        }
        with open(history_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history_header)
            writer.writerow(items)

    return max_reward
