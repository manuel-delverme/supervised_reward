import os
import random
import time

import numpy as np
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import tensorboardX
import torch.nn as nn
import torch.optim
import tqdm

import config
import shared.constants
import shared.utils


class CachedPolicy:
    def __init__(self, estimator, reward_function, reward_vector):
        self.estimator = estimator
        self.cache = {}
        self.reward_vector = reward_vector
        self.reward_matrix = reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size)
        self.reward_function = reward_function

    def __setitem__(self, obs_hash, value):
        obs_hash = shared.utils.hash_image(obs_hash)
        self.cache[obs_hash] = value

    def __getitem__(self, image):
        if self.reward_function(image.reshape(-1)) > 0:
            return -1

        obs_hash = shared.utils.hash_image(image)
        if obs_hash not in self.cache:
            q_values = self.estimator.predict(image)
            action_idx = int(np.argmax(q_values))
            self.cache[obs_hash] = action_idx
        return self.cache[obs_hash]

    def __str__(self):
        return self.reward_vector.__str__()

    def __repr__(self):
        return self.reward_vector.__str__()


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
        observations = np.array(observations).reshape(1, -1)
        assert observations.shape == (1, 196)
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)
        return self.featurizer.transform(observations.astype(np.float))


class LinearRegressionModel(nn.Module):
    def __init__(self, nr_inputs):
        super(LinearRegressionModel, self).__init__()
        self.regressor = nn.Linear(nr_inputs, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        y_pred = self.regressor(x)
        return y_pred


class PytorchEstimator:
    def __init__(self, action_size: int, observation_size: int, nr_usable_actions):
        self.models = []
        self.optimizers = []
        self.criterion = nn.MSELoss()
        self.nr_inputs = observation_size
        self.nr_usable_actions = nr_usable_actions

        for _ in range(action_size):
            self.add_new_action()

    def add_new_action(self):
        model = LinearRegressionModel(nr_inputs=self.nr_inputs).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # TODO: init small weights

        self.models.append(model)
        self.optimizers.append(optimizer)

    def predict(self, observation, action=None):
        assert action is None or action > -1
        features = self.preprocess(observation)

        if not action:
            prediction = []
            for model in self.models:
                p = model(features)
                prediction.append(p[0])
            prediction = np.array(prediction)
        else:
            prediction = self.models[action](features)
        return prediction

    def update(self, observation, a, target):
        target = torch.FloatTensor([target]).unsqueeze(-1).to(config.device)

        optimizer = self.optimizers[a]
        model = self.models[a]
        features = self.preprocess(observation)

        # for _ in range(iters):
        optimizer.zero_grad()

        pred = model(features)
        loss = self.criterion(pred, target)
        loss.backward()

        optimizer.step()

    def preprocess(self, observation):
        # features = observation.reshape(1, -1)
        f = torch.from_numpy(observation).float()
        f = f.unsqueeze(0)
        f.flatten(start_dim=1)
        # f = f.permute(0, 3, 1, 2)
        f = f.to(config.device)
        return f

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


def pick_action_test(state, old_action_idx, old_primitive_action, is_option, available_actions=None, estimator=None):
    return pick_action(state, old_action_idx, old_primitive_action, is_option, available_actions=available_actions, estimator=estimator, exploit=True)


def pick_action(observation, old_action_idx, old_primitive_action, is_option, *, exploit=False, epsilon=None, available_actions=None, estimator=None):
    was_option = is_option(old_action_idx)
    # logger.add_scalar('learning/was_option', was_option)

    if not was_option or old_primitive_action == shared.constants.TERMINATE_OPTION:
        if exploit or epsilon < random.random():
            # greedy
            q_values = estimator.predict(observation)
            q_values = q_values[:len(available_actions)]
            action_idx = int(np.argmax(q_values))
        else:
            # explore
            action_idx = random.randint(0, len(available_actions) - 1)

            while is_option(action_idx) and available_actions[action_idx][observation] == shared.constants.TERMINATE_OPTION:
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


def learn(environment, *, options=False, epsilon=config.learn_epsilon, gamma=0.90, surrogate_reward=None, generate_options=False, terminate_on_surr_reward=False,
          training_steps=None, replace_reward=config.replace_reward, eval_fitness=True, option_nr=-1, log_postfix=''):
    position = 0
    if generate_options:
        type_of_run = 'discovery'
        position = 1
    else:
        if eval_fitness:
            type_of_run = 'eval'
            # position = 1
        else:
            type_of_run = 'option'
            # position = 1

    if log_postfix:
        logger = tensorboardX.SummaryWriter(os.path.join('runs', config.experiment_name, log_postfix), flush_secs=1)
    else:
        raise Exception
        logger = config.tensorboard

    nr_primitive_actions = environment.action_space.n
    available_actions = list(range(nr_primitive_actions))
    action_size = environment.action_space.n
    action_to_id = {k: k for k in range(action_size)}

    if options:
        available_actions.extend(options)
        action_size += len(options)
        for option in options:
            action_to_id[option] = len(action_to_id)

    nr_usable_actions = len(available_actions)  # disallow learned options to be used
    print('nr available actions', nr_usable_actions)
    estimator = PytorchEstimator(environment.action_space.n + len(options or []), observation_size=environment.observation_space.n, nr_usable_actions=nr_usable_actions)

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

    learned_options = set()
    terminal = False
    reward = None

    old_state = environment.reset()
    new_state = old_state

    if surrogate_reward is not None and (config.enjoy_surrogate_reward or (
            config.enjoy_option and not generate_options)):
        print('plotting', type_of_run)
        shared.utils.plot_surrogate_reward(environment, surrogate_reward)

        print('enjoying', type_of_run)
        shared.utils.enjoy_surrogate_reward(environment, surrogate_reward)

    desc = f"Training {type_of_run}, option nr {option_nr}"
    for step in tqdm.tqdm(range(training_steps), desc=desc, disable=config.disable_tqdm, position=position):

        action_idx, primitive_action = pick_action(
            observation=old_state, old_action_idx=action_idx, epsilon=epsilon, available_actions=available_actions[:nr_usable_actions], estimator=estimator,
            old_primitive_action=primitive_action, is_option=is_option)

        # This is super slow, activate if necessary
        # logger.add_histogram(f'learning{option_nr}/action_idx', action_idx, step)

        # logger.add_scalar(f'learning{option_nr}/distance_traveled', np.linalg.norm(environment.agent_position_idx - initial_position), step)
        # logger.add_scalar(f'learning{option_nr}/steps_left', environment.env.steps_remaining, step)

        if options:
            if is_option(action_idx):
                logger.add_scalar(f'learning{option_nr}/option_taken', 1, step)
                if option_begin_state is None:
                    option_begin_state = old_state

            else:
                logger.add_scalar(f'learning{option_nr}/option_taken', 0, step)

        # logger.add_scalar('debug/step', step, step)
        if primitive_action == shared.constants.TERMINATE_OPTION:
            # option update and state reset
            discounted_future_value = gamma * max(estimator.predict(new_state))
            td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, discounted_future_value)

            time_steps_under_option = 0
            discounted_reward_under_option = 0
            option_begin_state = None

        else:
            new_state, reward, terminal, info = environment.step(available_actions[primitive_action])
            if environment.ob_rms:
                logger.add_scalar(f'learning{option_nr}/obmean', environment.ob_rms.mean.mean(), step)
                logger.add_scalar(f'learning{option_nr}/obvar', environment.ob_rms.var.mean(), step)
            fitness += 1 if reward > 0 else 0

            if not replace_reward:
                logger.add_scalar(f'learning{option_nr}/real_reward', reward, step)
                logger.add_scalar(f'learning{option_nr}/fitness', fitness, step)

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
            logger.add_scalar(f'learning{option_nr}/received_reward', reward, step)

            cumulative_reward += reward
            # logger.add_scalar(f'learning{option_nr}/cum_received_reward', cumulative_reward, step)

            if type_of_run == 'eval' and config.enjoy_master_learning and step > (training_steps - 500):
                environment.render(reward=reward, step=step, action_idx=action_idx)
                # if action_idx >= nr_primitive_actions:
                #     input('>')
                # time.sleep(0.3)
                # if terminal:
                #     fancy_render(environment, estimator, seen_rewards, step)

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

            # logger.add_scalar(f'learning{option_nr}/action_td', reward + discounted_future_value, step)
            td_update(estimator, old_state, primitive_action, reward, discounted_future_value)

            if is_option(action_idx):
                assert option_begin_state is not None
                discounted_reward_under_option += reward * (gamma ** time_steps_under_option)
                time_steps_under_option += 1

                if terminal:
                    # logger.add_scalar(f'learning{option_nr}/option_td', reward + discounted_future_value, step)
                    td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, future_reward=0)
                    time_steps_under_option = 0
                    discounted_reward_under_option = 0
                    option_begin_state = None
                    primitive_action = shared.constants.TERMINATE_OPTION

        if generate_options and reward > config.option_trigger_treshold:
            option_nr = len(learned_options)

            motivating_function = np.multiply(surrogate_reward.reward_vector, new_state.reshape(-1))

            option_hash = hash_option(motivating_function)

            if option_hash not in learned_options:
                # reward = self.surrogate_reward(self.environment)
                # goal = self.environment.agent_position_idx

                # motivating_function = np.einsum('s,s->s', surrogate_reward.reward_vector, new_state.reshape(-1))

                # for _ in range(200):
                #     environment.render()
                #     time.sleep(0.01)

                new_option = learn_option(motivating_function, environment, option_nr=option_nr)
                learned_options.add(option_hash)

                available_actions.append(new_option)

                # learned options are not used
                # estimator.add_new_action()

            if option_nr >= config.max_nr_options and not eval_fitness:
                opts = available_actions[nr_primitive_actions:]
                return opts, cumulative_reward, None, estimator

        old_state = new_state

    if eval_fitness:
        test_fitness = test(
            environment=environment, estimator=estimator, eval_steps=config.option_eval_test_steps, render=False, terminate_on_surr_reward=terminate_on_surr_reward,
            is_option=is_option, surrogate_reward=surrogate_reward, available_actions=available_actions[:nr_usable_actions])
        # print("FITNESS: ", test_fitness)
        # test_fitness = test(environment=environment, estimator=estimator, eval_steps=config.option_eval_test_steps // 10, render=True,
        #                     terminate_on_surr_reward=terminate_on_surr_reward, is_option=is_option, surrogate_reward=surrogate_reward, available_actions=available_actions)
        # fancy_render(environment, estimator, seen_rewards, step)
    else:
        test_fitness = None

    opts = available_actions[nr_primitive_actions:]
    return opts, cumulative_reward, test_fitness, estimator


def hash_option(motivating_function):
    # TODO: this is an hack
    # option_hash = motivating_function.data.tobytes()
    option_hash, = np.where(motivating_function > config.option_trigger_treshold)
    option_hash = option_hash.data.tobytes()
    return option_hash


def fancy_render(environment, estimator, seen_rewards, step):
    import matplotlib.pyplot as plt

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
        if replace_reward:
            reward = surr_reward
        else:
            reward += surr_reward
        if terminate_on_surr_reward:
            terminal = surr_reward > 0
        else:
            terminal = reward > 0
    return reward, terminal


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


# @disk_utils.disk_cache
def learn_option(option_reward_vector, mdp, *, training_steps=config.option_train_steps, option_nr=-1):  # reduced for 7x7

    def option_reward(state, info=None):
        return option_reward_vector.dot(state.reshape(-1))

    option_reward.reward_vector = option_reward_vector

    _, _, _, estimator = learn(
        environment=mdp, options=None, surrogate_reward=option_reward, training_steps=training_steps, eval_fitness=False,
        terminate_on_surr_reward=True, replace_reward=True, option_nr=option_nr, log_postfix=f'learn_option{option_nr}')

    option = CachedPolicy(estimator, reward_function=option_reward, reward_vector=option_reward_vector)

    if config.enjoy_learned_options:
        print(f'learned option[{option_nr}] for GOAL:', option_reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size))

        mdp.render()
        mdp.render()
        shared.utils.enjoy_policy(mdp, option, option_reward)
    return option
