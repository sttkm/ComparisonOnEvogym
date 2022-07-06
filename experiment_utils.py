import os
import shutil
import json
import numpy as np


def initialize_experiment(experiment_name, save_path, args):
    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            quit()
        print()

    argument_file = os.path.join(save_path, 'arguments.json')
    with open(argument_file, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_experiment(expt_path):
    with open(os.path.join(expt_path, 'arguments.json'), 'r') as f:
        expt_args = json.load(f)
    return expt_args


from evogym import is_connected, has_actuator, get_full_connectivity

def load_robot(robot, task=None):

    if robot=='default':
        robot = task
        robot_file = os.path.join('robot_files', f'{robot}.txt')
        assert os.path.exists(robot_file), f'defalt robot is not set on the task {task}'
    else:
        robot_file = os.path.join('robot_files', f'{robot}.txt')

    robot = np.loadtxt(robot_file)
    assert is_connected(robot), f'robot {args.robot} is not fully connected'
    assert has_actuator(robot), f'robot {args.robot} have not actuator block'

    connectivity = get_full_connectivity(robot)
    structure = (robot, connectivity)

    return structure