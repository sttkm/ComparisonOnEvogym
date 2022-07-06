import os

import evogym.envs

from experiment_utils import initialize_experiment, load_robot

from utils.ppo import run_ppo
from utils.simulator import EvogymControllerSimulatorPPO, SimulateProcess


from arguments.evogym_ppo import get_args

class ppoConfig():
    def __init__(self, args):
        self.num_processes = args.num_processes
        self.eval_processes = 2
        self.seed = 1
        self.steps = args.steps
        self.num_mini_batch = args.num_mini_batch
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.clip_range = args.clip_range
        self.ent_coef = 0.01

        self.learning_steps = 50

        self.policy_kwargs = {
            'log_std_init'  : 0.0,
            'ortho_init'    : True,
            'squash_output' : False,
        }


def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_ppo', args.name)

    initialize_experiment(args.name, save_path, args)


    structure = load_robot(args.robot, task=args.task)

    ppo_config = ppoConfig(args)


    controller_path = os.path.join(save_path, 'controller')
    os.makedirs(controller_path, exist_ok=True)

    if args.view:
        simulator = EvogymControllerSimulatorPPO(
            env_id=args.task,
            structure=structure,
            load_path=controller_path,
            deterministic=not args.probabilistic)

        simulate_process = SimulateProcess(
            simulator=simulator,
            generations=args.ppo_iters)

        simulate_process.init_process()
        simulate_process.start()


    history_file = os.path.join(save_path, 'history.csv')
    run_ppo(
        env_id=args.task,
        structure=structure,
        train_iters=args.ppo_iters,
        config=ppo_config,
        save_path=controller_path,
        history_file=history_file,
        deterministic=not args.probabilistic,
        )

if __name__=='__main__':
    main()
