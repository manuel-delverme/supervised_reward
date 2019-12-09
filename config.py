import sys
import warnings

import torch

# this is used in  gym-minigrid/gym_minigrid/minigrid.py
# seed = 31337 too easy
seed = 1337

agent_view_size = 5  # 7
import envs.minigrid

environment = envs.minigrid.MiniGrid
# env_name = 'MiniGrid-MultiRoom-N2-S1-v0'
env_name = "MiniGrid-DoorKey-5x5-v0"
# env_name = "MiniGrid-DoorKey-6x6-v0"
# env_name = "MiniGrid-DoorKey-8x8-v0"
# env_name = "MiniGrid-DoorKey-16x16-v0"
# env_name = 'MiniGrid-MultiRoom-N2-S4-v0'

# env_name = 'MiniGrid-MultiRoom-N2-S6-v0'
# env_name = 'MiniGrid-MultiRoom-N6-v0' # impossible
# env_name = 'MiniGrid-Empty-6x6-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'

see_trough_walls = False
warnings.warn('SET TRUE ABOVE')
warnings.warn('check if walkable/nonwalkable is needded')

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
print("DEBUG: ", DEBUG)

device = torch.device("cpu")  # GPU is slower for small networks
print(f"USING {device}")

repeat_eval_options = 1 if DEBUG else 1

fitness_training_steps = None
eval_test_restarts = 1  # 0

option_discovery_steps = 10002
option_eval_test_steps = 10003

option_eval_training_steps = 20002
option_train_steps = 100005

evolution_iters = 150004
max_env_steps = 200

learning_rate = 1e-3
max_train_option_steps = 150
max_option_duration = 50

if 'S1' in env_name:
    print("DECIMATING TIMES BECAUSE S1")
    option_train_steps //= 10
    option_discovery_steps //= 5
    option_eval_training_steps //= 5
    max_env_steps //= 5  # None

if '16' in env_name:
    max_option_duration *= 2
    option_train_steps *= 2
    option_discovery_steps *= 2
    option_eval_training_steps *= 1
    max_env_steps *= 5  # None

# if DEBUG:
#     print("DECIMATING TIMES BECAUSE DEBUG")
#     eval_test_restarts = 1
#     option_train_steps //= 10
#     option_discovery_steps //= 100
#     option_eval_training_steps //= 10
#     option_eval_test_steps //= 10
#     evolution_iters = 10

# main.env_name = "debug"
# env_name = "boxes"
# env_name = "hungry-thirsty"
# env_name = "minigrid"

generate_on_rw = True
replace_reward = True
use_learned_options = False
shape_reward = False
compact_observation = False

visualize_all = 0
enjoy_surrogate_reward = visualize_all or 0
enjoy_motivating_function = visualize_all or 0

enjoy_master_learning = visualize_all or 0
enjoy_option_learning = visualize_all or 0

enjoy_learned_options = visualize_all or 0
# enjoy_option = visualize_all or 0
enjoy_test = visualize_all or 0

visualize_any = any((enjoy_surrogate_reward, enjoy_master_learning,
                     enjoy_option_learning, enjoy_learned_options,
                     # enjoy_option,
                     enjoy_test,))

disable_tqdm = False

population = 20


class Minigrid:
    gamma = 0.99
    epsilon = 1e-8
    nr_layers = 4


learn_epsilon = 0.1
max_nr_options = 1  # overwritten by mainLTL
option_trigger_treshold = 1.0
option_termination_treshold = option_trigger_treshold
BATCH_SIZE = 64

print({
    'option_discovery_steps': option_discovery_steps,
    'option_eval_test_steps': option_eval_test_steps,
    'option_eval_training_steps': option_eval_training_steps,
    'option_train_steps': option_train_steps,
    'max_train_option_steps': max_train_option_steps,
})

no_variance = True
trivial_observations = False  # not implemented
blurred_observations = True
recalculate_fitness = True
automatic_pickup = True
exploration_bonus = False
option_train_epochs = 1
deliberation_cost = 0.0
multiprocess = False
TensorBoard = True  # slow
NO_CACHE_ON_DEBUG = True
HACK = False
