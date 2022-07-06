import os

import evogym.envs

import neat_
from parallel import ParallelEvaluator
from experiment_utils import initialize_experiment, load_robot

from utils.evaluator import EvogymControllerEvaluator
from utils.simulator import EvogymControllerSimulator, SimulateProcess
from utils.gym_utils import make_vec_envs


from arguments.evogym_neat import get_args


def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_neat', args.name)

    initialize_experiment(args.name, save_path, args)


    structure = load_robot(args.robot, task=args.task)


    decode_function = neat_.FeedForwardNetwork.create

    evaluator = EvogymControllerEvaluator(args.task, structure, args.eval_num)
    evaluate_function = evaluator.evaluate_controller

    parallel = ParallelEvaluator(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )


    env = make_vec_envs(args.task, structure, 0, 1)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    env.close()

    config_file = os.path.join('config', 'evogym_neat.cfg')
    custom_config = [
        ('NEAT', 'pop_size', args.pop_size),
        ('DefaultGenome', 'num_inputs', num_inputs),
        ('DefaultGenome', 'num_outputs', num_outputs)
    ]
    config = neat_.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'evogym_neat.cfg')
    config.save(config_out_file)


    pop = neat_.Population(config)

    reporters = [
        neat_.SaveResultReporter(save_path),
        neat_.StdOutReporter(True),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)


    if args.view:
        simulator = EvogymControllerSimulator(
            env_id=args.task,
            structure=structure,
            decode_function=decode_function,
            load_path=save_path,
            history_file='history_reward.csv',
            genome_config=config.genome_config)

        simulate_process = SimulateProcess(
            simulator=simulator,
            generations=args.generation)

        simulate_process.init_process()
        simulate_process.start()

    pop.run(fitness_function=parallel.evaluate, n=args.generation)

if __name__=='__main__':
    main()
