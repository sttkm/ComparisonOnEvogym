# Comparison of algorithms for behabioral learning on Evolution Gym

## abstract
[Evolution Gym](https://evolutiongym.github.io/)で、ロボットの制御学習のためのアルゴリズムを比較する。
Evolution Gymには32のタスクが用意されており、それらに指定したロボットの構造の制御方法を比較することができる。  
制御方法を学習するアルゴリズムが、タスクとロボット構造の組み合あわせに対して、どのように働くかを実験することを目的として作成した。学習アルゴリズムとしてNEAT、HyperNEAT、PPOの3つを提供し、全てニューラルネットワークによってロボットを制御する。

### samples
以下に、各アルゴリズムの実験で得られた結果のサンプルを示す。
- task: Walker-v0  
  - algo: NEAT  
![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Walker-v0_ppo-opt_neat.gif)
  - algo: HyperNEAT  
![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Walker-v0_ppo-opt_hyper.gif)
  - algo: PPO  
![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Walker-v0_ppo-opt_ppo.gif)

- task: PlatformJumper-v0
  - algo: NEAT  
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/PlatformJumper-v0_hand_neat.gif)
  - algo: HyperNEAT  
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/PlatformJumper-v0_hand_hyper.gif)
  - algo: PPO  
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/PlatformJumper-v0_hand_ppo.gif)

- task: Climber-v0
  - algo: NEAT, HyperNEAT, PPO  
  <img src="https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Climber-v0_hand_neat.gif" width="10%" height="10%"> <img src="https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Climber-v0_hand_hyper.gif" width="10%" height="10%"> <img src="https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Climber-v0_hand_ppo.gif" width="10%" height="10%">

- task: Flipper-v0
  - algo: NEAT
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Flipper-v0_ppo-opt_neat.gif)
  - algo: HyperNEAT
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Flipper-v0_ppo-opt_hyper.gif)
  - algo: PPO
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Flipper-v0_ppo-opt_ppo.gif)

- task: Thrower-v0
  - algo: NEAT
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Thrower-v0_hand_neat.gif)
  - algo: HyperNEAT
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Thrower-v0_hand_hyper.gif)
  - algo: PPO
  ![](https://github.com/sttkm/ComparisonOnEvogym/blob/main/gifs/Thrower-v0_hand_ppo.gif)




## Experiment

### requrements
- Python 3.8.12
- [Evolution Gym](https://evolutiongym.github.io/tutorials/getting-started.html)

### NEAT
#### execution
```
$python run_neat.py
```
##### options:
| option      | abbrev  | default         | detail  |
| :---        | :---:   | :---:           | :---    |
| --name      | -n      | "{task}_{robot}"| experiment name |
| --task      | -t      | Walker-v0       | evogym environment id |
| --robot     | -r      | default         | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --pop-size  | -p      | 200             | population size of NEAT |
| --generation| -g      | 500             | iterations of NEAT |
| --eval-num  |         | 1               | evaluation times. if probabilistic task, need more. |
| --num-cores | -c      | 4               | number of parallel evaluation processes |
| --view      |         | *false*         | open simulation window of best robot |


#### make gif
after run_neat, make gif file for each of all genomes written in reward history file.
output to "./out/evogym_neat/{expt name}/gif/"
```
$python make_gifs_neat.py {experiment name}
```
##### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input id, make figure for the only specified genome |
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |


### HyperNEAT
#### execution
```
$python run_hyper.py
```
##### options:
| option      | abbrev  | default         | detail  |
| :---        | :---:   | :---:           | :---    |
| --name      | -n      | "{task}_{robot}"| experiment name |
| --task      | -t      | Walker-v0       | evogym environment id |
| --robot     | -r      | default         | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --pop-size  | -p      | 200             | population size of NEAT |
| --generation| -g      | 500             | iterations of NEAT |
| --no-hideen |         | *false*         | not make hidden nodes on NN substrate |
| --eval-num  |         | 1               | evaluation times. if probabilistic task, need more. |
| --num-cores | -c      | 4               | number of parallel evaluation processes |
| --view      |         | *false*         | open simulation window of best robot |


#### make gif
after run_hyper, make gif file for each of all genomes written in reward history file.
output to "./out/evogym_hyper/{expt name}/gif/"
```
$python make_gifs_hyper.py {experiment name}
```
##### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input id, make figure for the only specified genome |
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |



### PPO
#### execution
```
$python run_ppo.py
```
##### options:
| option          | abbrev  | default         | detail  |
| :---            | :---:   | :---:           | :---    |
| --name          | -n      | "{task}_{robot}"| experiment name |
| --task          | -t      | Walker-v0       | evogym environment id |
| --robot         | -r      | default         | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --num-processes | -p      | 4               | how many training CPU processes to use |
| --steps         | -s      | 128             | num steps to use in PPO |
| --num-mini-batch| -b      | 4               | number of batches for ppo |
| --epochs        | -e      | 4               | number of ppo epochs |
| --ppo-iters     | -i      | 100             | learning iterations of PPO |
| --lerning-rate  | -lr     | 2.5e-4          | learning rate |
| --gamma         |         | 0.99            | discount factor for rewards |
| --clip-range    | -c      | 0.1             | ppo clip parameter |
| --probabilistic | -d      | *false*         | robot act probabilistic |
| --view          |         | *false*         | open simulation window of best robot |


#### make figure
after run_ppo, make gif file for each of all controllers.
output to "./out/evogym_ppo/{expt name}/gif/"
```
$python make_gifs_ppo.py {experiment name}
```
##### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input iter, make figure for the only specified controller |
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |
