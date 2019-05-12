import os
import sys
import time

import tensorboardX
import torch

agent_view_size = 5  # 7
import envs.minigrid

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
# DEBUG = False
print('DEBUG:', DEBUG)

environment = envs.minigrid.MiniGrid
# env_name = 'MiniGrid-Empty-6x6-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'
env_name = 'MiniGrid-MultiRoom-N2-S4-v0'
# env_name = 'MiniGrid-MultiRoom-N2-S6-v0'
# env_name = 'MiniGrid-MultiRoom-N6-v0' # impossible

population = 5

experiment_id = 'DEBUG'
if not DEBUG:
    try:
        import tkinter.simpledialog

        # comment = "256h32bs"
        root = tkinter.Tk()
        experiment_id = tkinter.simpledialog.askstring("comment", "comment")
        root.destroy()
    except Exception:
        experiment_id = 'comment_failed'

repeat_eval_options = 1 if DEBUG else 1

if DEBUG:
    eval_test_restarts = 1
    training_steps = 400
    option_discovery_steps = -1
    option_eval_training_steps = 10000
    option_eval_test_steps = 200
    evolution_iters = 10
    experiment_id += 'DEBUG'
    fitness_training_steps = None
else:
    fitness_training_steps = None
    eval_test_restarts = 1  # 0
    # training_steps = 1000  # 0
    option_discovery_steps = 10000
    option_eval_training_steps = 50000
    option_eval_test_steps = 1000
    evolution_iters = 100000

# main.env_name = "debug"
# env_name = "boxes"
# env_name = "hungry-thirsty"
# env_name = "minigrid"
tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_id, time.strftime("%Y_%m_%d-%H_%M_%S")), flush_secs=0.1)
tensorboard.add_custom_scalars({
    'best': {'optimization': ['Multiline',
                              ['optimization/baseline', 'optimization/best', 'optimization/random_best']]},
    'mean': {'optimization': ['Multiline',
                              ['optimization/baseline', 'optimization/mean', 'optimization/random_mean']]},
})

option_train_steps = 10 if DEBUG else 10000
learning_rate = 0.001

generate_on_rw = True
replace_reward = False
use_learned_options = False
visualize_learning = True
shape_reward = False
max_env_steps = None
device = torch.device('cuda:0')
compact_observation = False
