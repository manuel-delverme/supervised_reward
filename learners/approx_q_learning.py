import random
import time

import numpy as np
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import tqdm

import config
from utils import disk_utils

TERMINATE_OPTION = -1


class CachedPolicy:
    def __init__(self, estimator):
        self.estimator = estimator
        self.cache = {}

    def __setitem__(self, key, value):
        self.cache[tuple(key)] = value

    def __getitem__(self, item):
        if item not in self.cache:
            q_values = self.estimator.predict(item)
            action_idx = int(np.argmax(q_values))
            self.cache[item] = action_idx
        return self.cache[item]

    def __str__(self):
        if not hasattr(self, 'name'):
            for k, v in self.cache.items():
                if v == -1:
                    self.name = str(k)
                    break
        return 'GoTo{' + self.name + '}'

    def __repr__(self):
        return self.__str__()


class FeatureTransformer:
    def __init__(self):
        featurizer = sklearn.pipeline.FeatureUnion([
            # ("norm", sklearn.preprocessing.StandardScaler(copy=False)),
            ("rbf0", sklearn.kernel_approximation.RBFSampler(gamma=8.0, n_components=100)),
            ("rbf1", sklearn.kernel_approximation.RBFSampler(gamma=4.0, n_components=100)),
            ("rbf2", sklearn.kernel_approximation.RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", sklearn.kernel_approximation.RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", sklearn.kernel_approximation.RBFSampler(gamma=0.5, n_components=100)),
        ])
        self.featurizer = featurizer
        # obss = list(itertools.product(range(6), range(6), range(4)))
        # observations = np.array(obss).astype(np.float)
        self.featurizer.fit(np.random.randn(1, 147))

    def transform(self, observations):
        observations = np.array(observations).ravel().reshape(1, -1)
        assert observations.shape == (1, 196)
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)
        return self.featurizer.transform(observations.astype(np.float))


class Estimator:
    def __init__(self, action_size: int, alpha: float):
        self.models = []
        self.not_trained = []
        # self.feature_transformer = FeatureTransformer()
        self.alpha = alpha

        for _ in range(action_size):
            self.add_new_action()

    def add_new_action(self):
        model = sklearn.linear_model.SGDRegressor(learning_rate='constant', eta0=self.alpha)
        self.models.append(model)
        self.not_trained.append(True)

    def predict(self, observation, action=None):
        assert action is None or action > -1
        features = self.preprocess(observation)

        if (action is None and any(self.not_trained)) or (action is not None and not self.not_trained[action]):
            return [-random.random() / 1000 for _ in self.models]
        if not action:
            prediction = []
            for m in self.models:
                p = m.predict(features)
                prediction.append(p[0])
            return np.array(prediction)
        else:
            return self.models[action].predict(features)[0]

    def update(self, observation, a, target):
        iters = 10 if target > 0.1 else 1
        for _ in range(iters):
            features = self.preprocess(observation)
            self.models[a].partial_fit(features, [target])
            if self.not_trained[a]:
                self.models[a].coef_ = (np.random.randn(*self.models[a].coef_.shape) * 0.0001)
            self.not_trained[a] = False

    def preprocess(self, observation):
        # features = np.array(observation).reshape(1, -1)
        features = observation.reshape(1, -1)
        # features = self.feature_transformer.transform(observation)
        assert len(features.shape) == 2  # reshape(1, -1)
        return features

    # end update

    def sample_action(self, s, eps):
        # eps = 0
        # Technically, we don't need to do epsilon-greedy
        # because SGDRegressor predicts 0 for all states
        # until they are updated. This works as the
        # 'Optimistic Initial Values' method, since all
        # the rewards for Mountain Car are -1
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
    # end sample_action


# class ApproxQLearning(object):
#     def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.2, surrogate_reward=None, goal=None):

def pick_action_test(state, old_action_idx, old_primitive_action, is_option, nr_primitive_actions=None, available_actions=None, estimator=None):
    return pick_action(state, old_action_idx, old_primitive_action, is_option, nr_primitive_actions=nr_primitive_actions, available_actions=available_actions, estimator=estimator,
                       exploit=True)


def pick_action(observation, old_action_idx, old_primitive_action, is_option, *, exploit=False, epsilon=None, nr_primitive_actions=None, available_actions=None, estimator=None):
    was_option = is_option(old_action_idx)
    config.tensorboard.add_scalar('learning/was_option', was_option)

    if not was_option or old_primitive_action == TERMINATE_OPTION:
        if exploit or epsilon < random.random():
            # greedy
            q_values = estimator.predict(observation)
            action_idx = int(np.argmax(q_values))
        else:
            # explore
            action_idx = random.randint(0, len(available_actions) - 1)
            while is_option(action_idx) and available_actions[action_idx][observation] == TERMINATE_OPTION:
                action_idx = random.randint(0, len(available_actions) - 1)

        if is_option(action_idx):
            primitive_action_idx = available_actions[action_idx][observation]
        else:
            primitive_action_idx = action_idx
    else:
        # keep following the option
        assert was_option
        primitive_action_idx = available_actions[old_action_idx][observation]
        action_idx = old_action_idx
    return action_idx, primitive_action_idx


def td_update(estimator, state, action_idx, reward, future_reward):
    # q(s,a) = q(s,a) + alpha * [ R + gamma * max(Q(s',a)-Q(s,a)]
    td_target = reward + future_reward
    estimator.update(state, action_idx, td_target)
    # discounted_reward = discounted_reward_under_option + reward * (gamma ** time_difference)


def learn(environment, *, options=None, epsilon=0.1, gamma=0.90, alpha=config.alpha, surrogate_reward=None, goal=None, generate_options=False, terminate_on_surr_reward=False,
          training_steps=None,
          replace_reward=config.replace_reward, generate_on_rw=config.generate_on_rw, use_learned_options=config.use_learned_options, eval_fitness=True, ):
    estimator = Estimator(environment.action_space.n + len(options or []), alpha)
    nr_primitive_actions = environment.action_space.n
    available_actions = list(range(nr_primitive_actions))
    action_size = environment.action_space.n
    action_to_id = {k: k for k in range(action_size)}

    if options and options is not None:
        available_actions.extend(options)
        action_size += len(options)
        for option in options:
            action_to_id[option] = len(action_to_id)

    surrogate_reward = surrogate_reward

    def make_is_option(num_actions):
        def is_option(action):
            return action is not None and action >= num_actions

        return is_option

    is_option = make_is_option(nr_primitive_actions)

    # before removal: 1.[20-40]/it at 40
    if not generate_options and nr_primitive_actions == len(available_actions):
        is_option = lambda x: False

    cumulative_reward = 0
    fitness = 0
    time_steps_under_option = 0
    discounted_reward_under_option = 0

    option_begin_state = None
    action_idx = None
    primitive_action = None

    option_goals = set()
    terminal = False
    reward = None
    # seen_rewards = dict()

    old_state = environment.reset()
    initial_position = environment.agent_position_idx
    new_state = old_state

    # for step in range(training_steps):
    for step in tqdm.tqdm(range(training_steps), desc="Training"):
        action_idx, primitive_action = pick_action(old_state, old_action_idx=action_idx, epsilon=epsilon, nr_primitive_actions=nr_primitive_actions,
                                                   available_actions=available_actions, estimator=estimator, old_primitive_action=primitive_action, is_option=is_option)
        config.tensorboard.add_histogram('learn/action_idx', action_idx, step)
        config.tensorboard.add_scalar('learn/distance_traveled', np.linalg.norm(environment.agent_position_idx - initial_position), step)
        config.tensorboard.add_scalar('learn/steps_left', environment.env.steps_remaining, step)

        if is_option(action_idx):
            config.tensorboard.add_scalar('learn/option_taken', 1, step)
            if option_begin_state is None:
                option_begin_state = old_state

        else:
            config.tensorboard.add_scalar('learn/option_taken', 0, step)

        if config.visualize_learning and step > (training_steps - 100):
            environment.render(reward=reward, step=step)
            time.sleep(0.3)
            # if terminal:
            #     fancy_render(environment, estimator, seen_rewards, step)

        # config.tensorboard.add_scalar('debug/step', step, step)
        if primitive_action == TERMINATE_OPTION:
            # option update and state reset
            discounted_future_value = gamma * max(estimator.predict(new_state))
            td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, discounted_future_value)

            time_steps_under_option = 0
            discounted_reward_under_option = 0
            option_begin_state = None

        else:
            new_state, reward, terminal, info = environment.step(available_actions[primitive_action])
            fitness += 1 if reward > 0 else 0

            if config.shape_reward:
                next_cells = [
                    # *environment.env.doors,
                    environment.env.goal_pos
                ]

                distance = 10000
                for cell_pos in next_cells:
                    # cell = environment.env.grid.get(*cell_pos)
                    cell_distance = np.linalg.norm(environment.agent_position_idx - np.array(cell_pos))
                    distance = min(distance, cell_distance)

                reward -= distance / 100


            reward, terminal = update_reward(info, new_state, replace_reward, reward, terminal, terminate_on_surr_reward, surrogate_reward)

            cumulative_reward += reward
            config.tensorboard.add_scalar('learning/reward', reward, step)
            config.tensorboard.add_scalar('learning/cum_reward', cumulative_reward, step)

            if terminal:
                environment.reset()

            # ---- TD Update ----
            # max(Q(s',a)

            # at every time step
            # if option:
            # 1) discount the option reward counter by gamma
            # 2) 1 step learn the primitive action
            # 3) TODO: 1 step learn the option

            # gamma * max(Q(s',a)-Q(s,a)
            if terminal:
                discounted_future_value = 0
            else:
                discounted_future_value = gamma * max(estimator.predict(new_state))

            config.tensorboard.add_scalar('learning/action_td', reward + discounted_future_value, step)
            td_update(estimator, old_state, primitive_action, reward, discounted_future_value)

            if is_option(action_idx):
                assert option_begin_state is not None
                discounted_reward_under_option += reward * (gamma ** time_steps_under_option)
                time_steps_under_option += 1

                if terminal:
                    config.tensorboard.add_scalar('learning/option_td', reward + discounted_future_value, step)
                    td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, future_reward=0)
                    time_steps_under_option = 0
                    discounted_reward_under_option = 0
                    option_begin_state = None
                    primitive_action = TERMINATE_OPTION

        if generate_on_rw:
            option_gen_metric = reward
        else:
            option_gen_metric = delta_Q

        if generate_options and option_gen_metric > 0 and tuple(old_state) not in option_goals:
            # reward = self.surrogate_reward(self.environment)
            # goal = self.environment.agent_position_idx
            new_option = generate_option(environment, old_state, use_learned_options)
            option_goals.add(old_state)
            available_actions.append(new_option)
            estimator.add_new_action()

        old_state = new_state

    if eval_fitness:
        test_fitness = test(environment=environment, estimator=estimator, eval_steps=config.option_eval_test_steps, render=False,
                            terminate_on_surr_reward=terminate_on_surr_reward, is_option=is_option, surrogate_reward=surrogate_reward, available_actions=available_actions)
        # print("FITNESS: ", test_fitness)
        # test_fitness = test(environment=environment, estimator=estimator, eval_steps=config.option_eval_test_steps // 10, render=True,
        #                     terminate_on_surr_reward=terminate_on_surr_reward, is_option=is_option, surrogate_reward=surrogate_reward, available_actions=available_actions)
        # fancy_render(environment, estimator, seen_rewards, step)
    else:
        test_fitness = None

    opts = available_actions[environment.action_space.n:]
    return opts, cumulative_reward, test_fitness, estimator


def fancy_render(environment, estimator, seen_rewards, step):
    import matplotlib.pyplot as plt
    # environment.render()

    value_map = np.zeros((environment.env.height, environment.env.width))
    reward_map = np.zeros((environment.env.height, environment.env.width))
    # Render the grid
    for column in range(environment.env.width):
        for row in range(environment.env.height):
            # x, y, direction
            states = [(column, row, d) for d in range(4)]
            values = estimator.predict(states)
            value_map[row][column] = max(values)

            if seen_rewards is not None:
                state_rewards = [-0.001]
                for state in states:
                    if state in seen_rewards:
                        state_rewards.append(seen_rewards[state])
                reward_map[row][column] = sum(state_rewards)

    fig = plt.figure(1)
    ax = fig.add_subplot(211)
    plt.title('Value at step {}'.format(step))
    cax = ax.matshow(value_map)
    fig.colorbar(cax)

    if seen_rewards is not None:
        ax = fig.add_subplot(212)
        cux = ax.matshow(reward_map)
        fig.colorbar(cux)
    plt.show()


def update_reward(info, new_state, replace_reward, reward, terminal, terminate_on_surr_reward, surrogate_reward):
    if surrogate_reward is not None:
        surr_reward = surrogate_reward(new_state, info)
        if terminate_on_surr_reward:
            terminal = surr_reward > 0
        else:
            terminal = reward > 0

        if replace_reward:
            reward = surr_reward
        else:
            reward += surr_reward

    return reward, terminal


def generate_option(environment, goal, use_learned_options):
    new_option = learn_option(goal, environment)
    if use_learned_options:
        # new_option = learn_option(old_state, self.environment)
        option_idx = self.Q.shape[1] + 1
        self.action_to_id[new_option] = option_idx - 1
        tmp_Q = np.empty((self.Q.shape[0], option_idx))
        tmp_Q[:, :-1] = self.Q
        self.Q = tmp_Q
        self.Q[:, -1] = self.Q[:, :-1].mean(axis=1)
    return new_option


def test(environment, eval_steps, *, estimator=None, render=False, terminate_on_surr_reward=False, is_option=None, surrogate_reward=None, available_actions=None):
    cumulative_reward = 0
    fitness = 0
    primitive_action_idx = None
    action_idx = None

    old_state = environment.reset()

    for step in range(eval_steps):
        action_idx, primitive_action_idx = pick_action_test(
            old_state, available_actions=available_actions, old_primitive_action=primitive_action_idx, old_action_idx=action_idx, is_option=is_option, estimator=estimator)

        if primitive_action_idx == -1:
            continue

        new_state, reward, terminal, info = environment.step(available_actions[primitive_action_idx])
        cumulative_reward += reward
        fitness += 1 if reward > 0 else 0

        if surrogate_reward is not None:
            surr_reward = surrogate_reward(new_state, info)

            if terminate_on_surr_reward:
                terminal = surr_reward > 0
            else:
                terminal = reward > 0

        if terminal:
            environment.reset()

        if render:
            if action_idx > environment.action_space.n:
                goal = available_actions[action_idx].index(-1)
            else:
                goal = None
            fancy_render(environment, estimator, None, step)
            time.sleep(5. / 30.)

        old_state = new_state
    return fitness


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


@disk_utils.disk_cache
def learn_option(goal, mdp, training_steps=config.option_train_steps):  # reduced for 7x7
    print("generating policy for goal:{}".format(goal))
    goal = tuple(goal)

    def surrogate_reward(state, info):
        if goal == state:
            return 1
        else:
            dist = abs(goal[0] - state[0]) + abs(goal[1] - state[1]) + abs(goal[2] - state[2]) * 0.01
            return -dist / 10

    _, _, fitnesses, estimator = learn(environment=mdp, options=None, surrogate_reward=surrogate_reward, goal=goal, training_steps=training_steps, terminate_on_surr_reward=True,
                                       replace_reward=True)

    print('GOAL:', goal)
    # learner.test(1000, render=True, terminate_on_surr_reward=True)

    option = CachedPolicy(estimator)
    option[goal] = -1
    option.name = str(goal)
    return option
