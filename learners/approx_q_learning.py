import collections
import copy
import math
import os
import random
import time

import numpy as np
import tensorboardX
import torch.nn as nn
import torch.optim
import tqdm

import config
import shared.constants
import shared.utils
from learners.helpers import CachedPolicy
from shared import disk_utils


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class FakeWriter():
    def add_scalar(self, *args, **kwargs):
        pass


class LinearRegressionModel(nn.Module):
    def __init__(self, nr_inputs):
        super(LinearRegressionModel, self).__init__()
        # self.preprocc = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=1)
        self.regressor = nn.Sequential(
            # nn.Linear(nr_inputs, 1, bias=True),
            nn.Linear(nr_inputs, 1, bias=True),
            # nn.Linear(nr_inputs, 1, bias=True),
            # nn.ReLU(),
            # nn.Linear(128, 1),
        )
        for p in self.regressor.parameters():
            p.data *= 0.01

    def forward(self, x):
        h = x.flatten(start_dim=1)
        # h = self.preprocc(x)
        # h = h.flatten(start_dim=1)
        y_pred = self.regressor(h)
        return y_pred


class PytorchEstimator:

    def __init__(self, action_size: int, observation_size: int, nr_usable_actions):
        self.models = []
        self.optimizers = []
        # self.memory = ReplayMemory(1000)

        self.feature_buffer = collections.deque(maxlen=1)
        self.target_buffer = collections.deque(maxlen=1)

        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.nr_inputs = observation_size
        self.nr_usable_actions = nr_usable_actions

        for _ in range(action_size):
            self.add_new_action()

    def add_new_action(self):
        model = LinearRegressionModel(nr_inputs=self.nr_inputs).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, nesterov=True, momentum=0.1)

        self.models.append(model)
        self.optimizers.append(optimizer)

    def predict(self, observation, action=None):
        assert action is None or action > -1
        features = self.preprocess(observation)

        if not action:
            prediction = []
            for model in self.models:
                p = model(features).detach().numpy()
                prediction.append(p[0])

            prediction = np.array(prediction)
        else:
            prediction = self.models[action](features)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape[1] == 1
        assert prediction.shape[0] >= 4
        return prediction

    def update(self, observation, a, target, logger):
        # assert isinstance(target, list)
        # assert all(isinstance(t, np.ndarray) for t in target)
        # assert all(t.shape == (1,) for t in target)
        assert isinstance(target, (np.float64, np.float))
        assert not hasattr(target, 'shape') or target.shape == tuple()
        features = self.preprocess(observation)
        # target = torch.from_numpy(target)

        self.target_buffer.append(target)
        self.feature_buffer.append(features)

        if len(self.target_buffer) < self.target_buffer.maxlen:
            return

        optimizer = self.optimizers[a]
        model = self.models[a]

        features = torch.cat(list(self.feature_buffer))
        target = torch.tensor(list(self.target_buffer)).unsqueeze(-1)

        for _ in range(self.feature_buffer.maxlen):
            optimizer.zero_grad()
            pred = model(features)
            loss = self.criterion(pred, target)
            loss.backward()

            optimizer.step()

        self.feature_buffer.clear()
        self.target_buffer.clear()

        return loss.detach().numpy()

    def preprocess(self, observation):
        features = torch.from_numpy(observation).float()
        features = features.unsqueeze(0)  # add batch size
        return features

    # def preprocess(self, observation):
    #     # features = observation.reshape(1, -1)
    #     f = torch.from_numpy(observation).float()
    #     f = f.unsqueeze(0)
    #     f.flatten(start_dim=1)
    #     # f = f.permute(0, 3, 1, 2)
    #     f = f.to(config.evice)
    #     return f

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


def pick_action_test(state, old_action_idx, old_primitive_action, is_option, environment, available_actions=None, estimator=None):
    return pick_action(state, old_action_idx, old_primitive_action, is_option, environment, available_actions=available_actions, estimator=estimator, exploit=True)


def pick_action(observation, old_action_idx, old_primitive_action, is_option, environment, *, exploit=False, epsilon=None, available_actions=None, estimator=None,
                time_steps_under_option=0):
    was_option = is_option(old_action_idx)

    if was_option and old_primitive_action != shared.constants.TERMINATE_OPTION and time_steps_under_option < config.max_option_duration:
        # keep following the option
        primitive_action_idx = available_actions[old_action_idx].get_or_terminate(observation, environment)
        action_idx = old_action_idx
        action_value = [-1]
    else:
        if exploit or epsilon < random.random():
            # greedy
            q_values = estimator.predict(observation)
            q_values = q_values[:len(available_actions)]
            action_idx = int(np.argmax(q_values))
            # action_idx = np.random.choice(range(q_values.shape[0]), p=softmax(q_values[:, 0]))

            # best_action, second_best = np.argsort(q_values)[:2]
            # action_idx = int(best_action)

            # action_value = q_values[action_idx]  # - q_values[second_best]
            action_value = q_values
        else:
            # explore
            action_idx = random_sample(available_actions, is_option, observation, environment)
            action_value = [-1]

        if is_option(action_idx):
            primitive_action_idx = available_actions[action_idx].get_or_terminate(observation, environment)
            if primitive_action_idx == shared.constants.TERMINATE_OPTION:
                action_idx = random_sample(available_actions, is_option, observation, environment)
                if is_option(action_idx):
                    primitive_action_idx = available_actions[action_idx].get_or_terminate(observation, environment)
                else:
                    primitive_action_idx = action_idx
                action_value = [-1]

        else:
            primitive_action_idx = action_idx

    assert isinstance(primitive_action_idx, int) and not is_option(primitive_action_idx)
    return action_idx, primitive_action_idx, action_value


def random_sample(available_actions, is_option, observation, environment):
    action_idx = random.randint(0, len(available_actions) - 1)
    while is_option(action_idx) and available_actions[action_idx].get_or_terminate(observation, environment) == shared.constants.TERMINATE_OPTION:
        action_idx = random.randint(0, len(available_actions) - 1)
    return action_idx


def td_update(estimator, state, action_idx, reward, future_reward, logger):
    # q(s,a) = q(s,a) + alpha * [ R + gamma * max(Q(s',a)-Q(s,a)]
    td_target = reward + future_reward
    loss = estimator.update(state, action_idx, td_target, logger)
    # discounted_reward = discounted_reward_under_option + reward * (gamma ** time_difference)
    return loss


def learn(environment, *, options=(), epsilon=config.learn_epsilon, gamma=0.90, surrogate_reward=None, generate_options=False,
          training_steps=None, replace_reward=config.replace_reward, eval_fitness=True, option_nr=-1, log_postfix=''):
    if generate_options:
        type_of_run = 'discovery'
        position = 0
    else:
        if eval_fitness:
            type_of_run = 'eval'
            position = 0
        else:
            type_of_run = 'option'
            position = 0

    if config.DEBUG or config.TensorBoard:
        logger = tensorboardX.SummaryWriter(os.path.join('runs', config.experiment_name, log_postfix), flush_secs=1)
    else:
        logger = FakeWriter()  # experiment name is not set anyway

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
    # print('nr available actions', nr_usable_actions)
    estimator = PytorchEstimator(environment.action_space.n + len(options or []), observation_size=environment.observation_space.n, nr_usable_actions=nr_usable_actions)

    def make_is_option(num_actions):
        def is_option(action):
            return action is not None and action >= num_actions

        return is_option

    is_option = make_is_option(nr_primitive_actions)
    if not generate_options and nr_primitive_actions == len(available_actions):
        is_option = lambda x: False

    cumulative_reward = 0
    cumulative_real_reward = 0
    fitness = 0
    cum_loss = 0
    time_steps_under_option = 0
    discounted_reward_under_option = 0
    highest_reward = None
    rewards = collections.deque(maxlen=config.max_env_steps)
    convergences = collections.deque(maxlen=30)

    option_begin_state = None
    action_idx = None
    primitive_action = None
    learned_options = set()

    print("\nenjoy surrogate function")
    maybe_enjoy_surrogate(environment, surrogate_reward, type_of_run, available_actions[nr_primitive_actions:])
    desc = f"Training {type_of_run}, option nr {option_nr}"
    old_state, steps_since_last_restart = reset(available_actions, environment, nr_primitive_actions, surrogate_reward)
    new_state = old_state
    old_convergence = 0

    for step in tqdm.tqdm(range(training_steps), desc=desc, disable=config.disable_tqdm, position=position):
        action_idx, primitive_action, action_value = pick_action(
            observation=old_state, old_action_idx=action_idx, epsilon=epsilon, available_actions=available_actions, estimator=estimator,
            old_primitive_action=primitive_action, is_option=is_option, environment=environment, time_steps_under_option=time_steps_under_option)

        logger.add_scalar(f'learning{option_nr}/action_value', max(action_value), step)
        steps_since_last_restart += 1
        option_begin_state = option_prestep(action_idx, is_option, logger, old_state, option_begin_state, option_nr, options, step)

        if primitive_action == shared.constants.TERMINATE_OPTION:
            assert option_begin_state is not None, 'terminating an option which never began'
            discounted_reward_under_option, option_begin_state, time_steps_under_option = process_option(
                action_idx, discounted_reward_under_option, estimator, gamma, new_state, option_begin_state, time_steps_under_option, logger)

        else:
            new_state, reward, terminal, info = environment.step(available_actions[primitive_action])

            cumulative_real_reward = update_non_surrogate_metrics(cumulative_real_reward, fitness, logger, option_nr, reward, step)
            reward, terminal = update_reward(environment, new_state, replace_reward, reward, steps_since_last_restart, surrogate_reward, terminal, type_of_run,
                                             available_actions[nr_primitive_actions:])
            reward_improvement = 0
            if highest_reward is None:
                highest_reward = reward

            if reward > highest_reward:
                reward_improvement = reward - highest_reward
                highest_reward = reward

            logger.add_scalar(f'learning{option_nr}/cumulative_reward', cumulative_reward, step)
            logger.add_scalar(f'learning{option_nr}/surrogate_reward', reward, step)
            logger.add_scalar(f'learning{option_nr}/option_completed', reward >= config.option_termination_treshold, step)
            logger.add_scalar(f'learning{option_nr}/terminal', terminal, step)
            cumulative_reward += reward
            rewards.append(reward)

            maybe_render_train(action_idx, action_value, available_actions, environment, reward, step, terminal, training_steps, type_of_run, new_state)

            opt, available_actions, estimator, highest_reward = generate_new_option(estimator, available_actions, generate_options, learned_options, new_state, reward_improvement,
                                                                                    surrogate_reward, highest_reward, nr_primitive_actions)

            if len(available_actions[nr_primitive_actions:]) >= config.max_nr_options and not eval_fitness:
                return available_actions[nr_primitive_actions:], cumulative_reward, None, estimator

            if not is_option(action_idx):
                reward -= config.deliberation_cost

            # gamma * max(Q(s',a)-Q(s,a)
            if terminal:
                discounted_future_value = 0
            else:
                discounted_future_value = gamma * estimator.predict(new_state).max()

            loss = td_update(estimator, old_state, primitive_action, reward, discounted_future_value, logger)
            if loss is not None:
                cum_loss += loss
                logger.add_scalar(f'learning{option_nr}/cum_loss', cum_loss, step)
                logger.add_scalar(f'learning{option_nr}/loss', loss, step)

            if is_option(action_idx):
                assert option_begin_state is not None, f"is_option: {is_option(action_idx)}"
                discounted_reward_under_option += reward * (gamma ** time_steps_under_option)
                time_steps_under_option += 1

                if terminal:
                    # TODO: also train intermediate steps
                    for _ in range(config.option_train_epochs):
                        td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, future_reward=0, logger=logger)
                    time_steps_under_option = 0
                    discounted_reward_under_option = 0
                    option_begin_state = None
                    primitive_action = shared.constants.TERMINATE_OPTION

            if terminal:
                new_state, steps_since_last_restart = reset(available_actions, environment, nr_primitive_actions, surrogate_reward)

                convergence = sum(max(r, 0) for r in rewards)
                convergences.append(convergence)
                # logger.add_scalar(f'learning{option_nr}/delta_convergence', convergence - old_convergence, step)
                logger.add_scalar(f'learning{option_nr}/convergence', convergence, step)
                logger.add_scalar(f'learning{option_nr}/convergences', sum(convergences), step)
                # old_convergence = convergence

                if sum(convergences) == convergences.maxlen:
                    break

                rewards.clear()

        old_state = new_state

    if eval_fitness:
        # print('testing')
        with torch.no_grad():
            test_fitness = test(
                environment=environment, estimator=estimator, eval_steps=config.option_eval_test_steps, render=False, is_option=is_option,
                available_actions=available_actions[:nr_usable_actions],
                update_reward=lambda s, r, t: update_reward(environment, s, replace_reward, r, steps_since_last_restart, surrogate_reward, t, type_of_run,
                                                            available_actions[nr_primitive_actions:])
            )
            # print('tested')
    else:
        test_fitness = None

    opts = available_actions[nr_primitive_actions:]
    return opts, cumulative_reward, test_fitness, estimator


def maybe_enjoy_surrogate(environment, surrogate_reward, type_of_run, options):
    if type_of_run == "options":
        options = ()
    else:
        options = copy.deepcopy(options)

    reset(options, environment, 0, surrogate_reward)
    if (config.enjoy_surrogate_reward and type_of_run == "discovery") or (config.enjoy_motivating_function and type_of_run == "option"):
        shared.utils.enjoy_surrogate_reward(environment, surrogate_reward, inibited_rewards=options, type_of_run=type_of_run)


def reset(available_actions, environment, nr_primitive_actions, surrogate_reward):
    observation = environment.reset()
    if surrogate_reward is not None:
        surrogate_reward.reset()
    for available_action in available_actions[nr_primitive_actions:]:
        available_action.motivating_function.reset()
    return observation, 0


seen_states = {}


def update_reward(environment, new_state, replace_reward, reward, steps_since_last_restart, surrogate_reward, terminal, type_of_run, inibited_rewards):
    assert replace_reward
    if surrogate_reward is not None:
        reward = surrogate_reward(new_state, environment)
        if type_of_run in ('discovery', 'visualization'):
            for option in inibited_rewards:
                inibition = option.motivating_function(new_state, environment)
                reward -= inibition

    if type_of_run == 'option' and (steps_since_last_restart > config.max_train_option_steps or reward >= config.option_termination_treshold):
        terminal = True
    elif steps_since_last_restart > config.max_env_steps:
        terminal = True

    if config.exploration_bonus:
        tup = (tuple(environment.env.agent_pos))
        pre_count = 0
        if tup in seen_states:
            pre_count = seen_states[tup]

        # Update the count for this key
        new_count = pre_count + 1
        seen_states[tup] = new_count
        bonus = 1 / math.sqrt(new_count)
        reward += bonus

    return reward, terminal


def generate_new_option(estimator, available_actions, generate_options, learned_options, state, reward, surrogate_reward, highest_reward, nr_primitive_actions):
    new_option = False

    if generate_options and reward >= config.option_trigger_treshold:
        motivating_function = surrogate_reward.motivating_function(state)
        option_hash = hash_option(motivating_function)

        if option_hash not in learned_options:
            print("\ndiscovered new option", motivating_function, 'of', surrogate_reward, '\n')
            new_option = learn_option(motivating_function, available_actions, nr_primitive_actions, option_nr=len(learned_options))

            learned_options.add(option_hash)
            available_actions.append(copy.deepcopy(new_option))
            estimator.add_new_action()
            print("\nenjoy the new discovery reward")
            maybe_enjoy_surrogate(config.environment(), surrogate_reward, "discovery", available_actions[nr_primitive_actions:])

        highest_reward = 0
    return new_option, available_actions, estimator, highest_reward


def maybe_render_train(action_idx, action_value, available_actions, environment, reward, step, terminal, training_steps, type_of_run, new_state):
    if (
            type_of_run == 'eval' and config.enjoy_master_learning and step > (training_steps - 100)
    ) or (
            type_of_run == 'option' and config.enjoy_option_learning and step > (training_steps - 50)
    ):
        # action_names = ('<', '>', 'foward', 'toggle', *range(len(available_actions) - 4))
        # action_value_ = []
        # for z in zip(action_names, (str(float(a))[:5] for a in action_value)):
        #     action_value_.append(str(z))
        # action_names = ('value',)
        action_value_ = (
            f"maxQ={max(action_value)}",
            f"act={available_actions[action_idx]}",
        )

        environment.render(type_of_run=type_of_run, reward=reward, step=step, action_idx=action_idx, action_value=action_value_, observation=new_state)
        time.sleep(0.2)
        if terminal:
            time.sleep(0.5)


def update_non_surrogate_metrics(cumulative_real_reward, fitness, logger, option_nr, reward, step):
    fitness += 1 if reward > 0 else 0
    cumulative_real_reward += reward
    logger.add_scalar(f'learning{option_nr}/fitness', fitness, step)
    logger.add_scalar(f'learning{option_nr}/cumulative_real_reward', cumulative_real_reward, step)
    return cumulative_real_reward


def process_option(action_idx, discounted_reward_under_option, estimator, gamma, new_state, option_begin_state, time_steps_under_option, logger):
    # option update and state reset
    discounted_future_value = gamma * estimator.predict(new_state).max()
    td_update(estimator, option_begin_state, action_idx, discounted_reward_under_option, discounted_future_value, logger)
    time_steps_under_option = 0
    discounted_reward_under_option = 0
    option_begin_state = None
    return discounted_reward_under_option, option_begin_state, time_steps_under_option


def option_prestep(action_idx, is_option, logger, old_state, option_begin_state, option_nr, options, step):
    if is_option(action_idx):
        logger.add_scalar(f'learning{option_nr}/option_taken', 1, step)
        if option_begin_state is None:
            option_begin_state = old_state

    else:
        logger.add_scalar(f'learning{option_nr}/option_taken', 0, step)
    return option_begin_state


def hash_option(intrinsic_motivation):
    return repr(intrinsic_motivation)


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


def test(environment, eval_steps, *, estimator=None, render=False, is_option=None, available_actions=None, update_reward=None):
    cumulative_reward = 0
    fitness = 0
    primitive_action_idx = None
    action_idx = None

    old_state = environment.reset()
    skips = 0

    for step in range(eval_steps):
        action_idx, primitive_action_idx, action_value = pick_action_test(
            old_state, environment=environment, available_actions=available_actions, old_primitive_action=primitive_action_idx, old_action_idx=action_idx, is_option=is_option,
            estimator=estimator)

        if primitive_action_idx == -1:
            continue

        new_state, reward, terminal, info = environment.step(available_actions[primitive_action_idx])
        if config.enjoy_test:
            action_names = ('<', '>', 'foward', 'toggle', *range(len(available_actions) - 4))
            action_value_ = [str(z) for z in zip(action_names, (float(a) for a in action_value))]

            environment.render(type_of_run='test', reward=reward, step=step, action_idx=action_idx, primitive_action_idx=primitive_action_idx, action_value=action_value_)
            time.sleep(0.5)

        cumulative_reward += reward

        fitness += 1 if reward > 0 else 0
        reward, terminal = update_reward(new_state, reward, terminal)

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


@disk_utils.disk_cache
def learn_option(option_reward, available_actions, nr_primitive_actions, *, option_nr=-1):
    _, _, _, estimator = learn(
        environment=config.environment(), options=available_actions[nr_primitive_actions:], surrogate_reward=option_reward, training_steps=config.option_train_steps,
        eval_fitness=False, replace_reward=True, option_nr=option_nr, log_postfix=f'option_of_{option_reward}')
    option_reward.reset()
    option = CachedPolicy(copy.deepcopy(estimator), motivating_function=option_reward, available_actions=available_actions)
    if config.enjoy_learned_options:
        mdp = config.environment()
        with np.printoptions(precision=3, suppress=True):
            print(f'learned option[{option_nr}] for GOAL:', option_reward, "enjoy policy")

            mdp.render()
            mdp.render()
            shared.utils.enjoy_policy(mdp, option, available_actions, option_reward)
    return option
