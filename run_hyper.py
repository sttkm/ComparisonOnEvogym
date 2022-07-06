import os

import evogym.envs

import neat_
from parallel import ParallelEvaluator
from experiment_utils import initialize_experiment, load_robot

from utils.evaluator import EvogymControllerEvaluator
from utils.simulator import EvogymControllerSimulator, SimulateProcess
from utils.hyper_decoder import EvogymHyperDecoder
from utils.substrate import Substrate


from arguments.evogym_hyper import get_args


def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_hyper', args.name)

    initialize_experiment(args.name, save_path, args)


    structure = load_robot(args.robot, task=args.task)


    substrate = Substrate(args.task, structure[0])
    decoder = EvogymHyperDecoder(substrate, use_hidden=not args.no_hidden, activation='sigmoid')
    decode_function = decoder.decode

    evaluator = EvogymControllerEvaluator(args.task, structure, args.eval_num)
    evaluate_function = evaluator.evaluate_controller

    parallel = ParallelEvaluator(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )


    config_file = os.path.join('config', 'evogym_hyper.cfg')
    custom_config = [
        ('NEAT', 'pop_size', args.pop_size),
        ('DefaultGenome', 'num_inputs', decoder.input_dims),
        ('DefaultGenome', 'num_outputs', decoder.output_dims)
    ]
    config = neat_.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'evogym_hyper.cfg')
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
