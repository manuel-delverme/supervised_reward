import sys
import time

try:
    import tkinter.simpledialog
except ImportError:
    HASGUI = False
else:
    HASGUI = True

import os
import tensorboardX
import torch

agent_view_size = 5  # 7
import envs.minigrid

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
print("DEBUG: ", DEBUG)

experiment_name = "DEBUG:" + time.strftime("%Y_%m_%d-%H_%M_%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device}")

response = None
if not DEBUG and HASGUI:
    try:
        import tkinter.simpledialog

        # comment = "256h32bs"
        root = tkinter.Tk()
        response = tkinter.simpledialog.askstring("comment", "comment")
        root.destroy()
    except tkinter.TclError as _:
        pass

if len(sys.argv) > 1:
    response = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'

repeat_eval_options = 1 if DEBUG else 1


fitness_training_steps = None
eval_test_restarts = 1  # 0
option_discovery_steps = 1000
option_eval_training_steps = 10000
option_eval_test_steps = 1000
evolution_iters = 100000

if DEBUG:
    eval_test_restarts = 1
    option_discovery_steps //= 100
    option_eval_training_steps //= 1000
    option_eval_test_steps //= 1000
    evolution_iters = 10

# main.env_name = "debug"
# env_name = "boxes"
# env_name = "hungry-thirsty"
# env_name = "minigrid"
option_train_steps = 10 if DEBUG else 1000
learning_rate = 0.001

generate_on_rw = True
replace_reward = False
use_learned_options = False
shape_reward = False
max_env_steps = None
compact_observation = False

visualize_learning = False
disable_tqdm = True

environment = envs.minigrid.MiniGrid
# env_name = 'MiniGrid-Empty-6x6-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'
env_name = 'MiniGrid-MultiRoom-N2-S4-v0'
# env_name = 'MiniGrid-MultiRoom-N2-S6-v0'
# env_name = 'MiniGrid-MultiRoom-N6-v0' # impossible


population = 2

print('EXPERIMENT:', experiment_name)
tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name), flush_secs=1)
