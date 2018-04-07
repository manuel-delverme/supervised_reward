import pickle
import tqdm
import collections
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

with open("best_sets.pkl", "rb") as fin:
    best_sets = pickle.load(fin)

vmax = 0
for iter_budget, option_sets in tqdm.tqdm(best_sets.items()):
    X, Y = np.meshgrid(range(7), range(7))
    Z = np.zeros((7, 7))
    total = 0
    # for idx in range(49):
    for option_set in option_sets:
        goal_idxs, score = option_set
        for goal_idx in goal_idxs:
            if goal_idx is not None:
                x = goal_idx % 7
                y = goal_idx // 7
                Z[x][y] += 1
                total += 1
    vmax = max(vmax, (Z / total).max())

for iter_budget, option_sets in tqdm.tqdm(best_sets.items()):
    X, Y = np.meshgrid(range(7), range(7))
    Z = np.zeros((7, 7))
    total = 0
    # for idx in range(49):
    for option_set in option_sets:
        goal_idxs, score = option_set
        for goal_idx in goal_idxs:
            if goal_idx is not None:
                x = goal_idx % 7
                y = goal_idx // 7
                Z[x][y] += 1
                total += 1
    Z = np.rot90(Z / total)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(Z, interpolation='nearest', vmax=vmax, vmin=0, cmap='jet')
    fig.colorbar(cax)
    ax.set_title('Goal distribution for budget: {} iterations'.format(iter_budget))

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    plt.savefig("plots/best_sets{}.png".format(iter_budget))
    fig.clear()
# plt.show()
