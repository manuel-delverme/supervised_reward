import sys
import time
import warnings

try:
    import tkinter.simpledialog
except ImportError:
    HASGUI = False
else:
    HASGUI = True

import os
import tensorboardX
import torch

seed = 31337  # this is used in  gym-minigrid/gym_minigrid/minigrid.py
agent_view_size = 5  # 7
import envs.minigrid

environment = envs.minigrid.MiniGrid
# env_name = 'MiniGrid-MultiRoom-N2-S1-v0'
env_name = 'MiniGrid-MultiRoom-N2-S4-v0'

# env_name = 'MiniGrid-MultiRoom-N2-S6-v0'
# env_name = 'MiniGrid-MultiRoom-N6-v0' # impossible
# env_name = 'MiniGrid-Empty-6x6-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'

see_trough_walls = False
warnings.warn('SET TRUE ABOVE')
warnings.warn('check if walkable/nonwalkable is needded')

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
print("DEBUG: ", DEBUG)

experiment_name = "DEBUG:" + time.strftime("%Y_%m_%d-%H_%M_%S")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")  # GPU is slower for small networks

print(f"USING {device}")

response = None
if not DEBUG and HASGUI:
    try:
        # comment = "256h32bs"
        root = tkinter.Tk()
        response = tkinter.simpledialog.askstring("comment", "comment")
        root.destroy()
    except tkinter.TclError as _:
        pass
    else:
        if response is None:
            response = "DELETEME"
            # DEBUG = True

experiment_name = f'{response}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'
if len(sys.argv) > 1:
    experiment_name = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'

repeat_eval_options = 1 if DEBUG else 1

fitness_training_steps = None
eval_test_restarts = 1  # 0

option_discovery_steps = 10002
option_eval_test_steps = 5003
option_eval_training_steps = 20002
option_train_steps = 100005

evolution_iters = 150004
max_env_steps = 200
max_train_option_steps = 50
learning_rate = 1e-3

if 'S1' in env_name:
    print("DECIMATING TIMES BECAUSE S1")
    option_train_steps //= 10
    option_discovery_steps //= 5
    option_eval_training_steps //= 5
    max_env_steps /= 5  # None

if DEBUG:
    print("DECIMATING TIMES BECAUSE DEBUG")
    eval_test_restarts = 1
    option_train_steps //= 10
    option_discovery_steps //= 100
    option_eval_training_steps //= 10
    option_eval_test_steps //= 10
    evolution_iters = 10

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

enjoy_master_learning = visualize_all or 0
enjoy_option_learning = visualize_all or 0

enjoy_learned_options = visualize_all or 0
enjoy_option = visualize_all or 0
enjoy_test = visualize_all or 0

disable_tqdm = True

population = 2

print('EXPERIMENT:', experiment_name)
# class fake_writer:
#     def add_scalar(*args):
#         pass
# tensorboard = fake_writer()
tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name), flush_secs=1)


class Minigrid:
    gamma = 0.99
    epsilon = 1e-8
    nr_layers = 3


learn_epsilon = 0.1
max_nr_options = 2
option_trigger_treshold = 1.0
option_termination_treshold = option_trigger_treshold
max_option_duration = 5
BATCH_SIZE = 64

print({
    'option_discovery_steps': option_discovery_steps,
    'option_eval_test_steps': option_eval_test_steps,
    'option_eval_training_steps': option_eval_training_steps,
    'option_train_steps': option_train_steps,
    'max_train_option_steps': max_train_option_steps,
})

no_variance = True
