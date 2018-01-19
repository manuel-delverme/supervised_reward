import numpy as np
import envs.gridworld
import collections
import tqdm
import learners.policy_iter
import sys
import time
import disk_utils
# import envs.boxes
import envs.hungry_thirsty
import random
import sys

TERMINATE_OPTION = -1


class QLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.3, surrogate_reward=None, test_run=False,
                 learning_option=False, train_run=False):

        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.nr_primitive_actions = self.environment.action_space.n
        self.available_actions = list(range(self.environment.action_space.n))

        assert (train_run ^ test_run ^ learning_option)
        self.test_run = test_run
        self.train_run = train_run
        self.learning_option = learning_option

        action_size = env.action_space.n
        self.action_to_id = {k: k for k in range(action_size)}
        if options is not None:
            self.available_actions.extend(options)
            action_size += len(options)
            for option in options:
                # goal = np.argwhere(action == -1)[0]
                self.action_to_id[option] = len(self.action_to_id)

        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, action_size)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, action_size)

    def pick_action(self, state, old_action_idx, old_primitive_action, kill_option=False):
        if kill_option:
            primitive_action_idx = TERMINATE_OPTION
            action_idx = old_action_idx
            # print("kill_option")
        # if the option terminated OR i was using primitives
        elif not self.is_option(old_action_idx) or old_primitive_action == TERMINATE_OPTION:

            if self.epsilon < random.random():
                action_idx = np.argmax(self.Q1[state] + self.Q2[state])
                # print("policy pick", action_idx)
            else:
                action_idx = random.randint(0, self.nr_primitive_actions - 1)

                while self.is_option(action_idx) and self.available_actions[action_idx][state] == TERMINATE_OPTION:
                    action_idx = random.randint(0, self.nr_primitive_actions - 1)
                    # print("random pick", action_idx)

            if self.is_option(action_idx):
                primitive_action_idx = self.available_actions[action_idx][state]
            else:
                primitive_action_idx = action_idx
        else:
            # keep following the option
            assert self.is_option(old_action_idx)
            primitive_action_idx = self.available_actions[old_action_idx][state]
            action_idx = old_action_idx
            # print("following option", action_idx)

        assert action_idx >= 0
        # print("[", action_idx, primitive_action_idx, "]")
        return action_idx, primitive_action_idx

    def learn(self, steps_of_no_change=None, generate_options=False, max_steps=None):
        assert (steps_of_no_change is not None or max_steps is not None)

        cumulative_reward = 0
        time_steps_under_option = 0
        discounted_reward_under_option = 0
        # no_change = 0
        # max_no_change = 0

        # render = 0
        option_begin_state = None
        action_idx = None
        primitive_action = None
        old_state = self.environment.reset()
        new_state = old_state
        option_goals = set()

        for step in range(max_steps):
            action_idx, primitive_action = self.pick_action(
                old_state, old_action_idx=action_idx, old_primitive_action=primitive_action)

            if option_begin_state is None and self.is_option(action_idx):
                option_begin_state = old_state

            Q, q = (self.Q1, self.Q2) if random.random() > 0.5 else (self.Q2, self.Q1)

            if primitive_action != TERMINATE_OPTION:
                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action])

                if self.surrogate_reward is not None:
                    reward = self.surrogate_reward(self.environment)
                    # if reward > 0 and self.training_run:
                    #     terminal = True
                    # Gianluca: reset when option is learnt!
                    # ME: nope

                cumulative_reward += reward

            if primitive_action == TERMINATE_OPTION:
                # end of option
                # update the option value function where the option started

                assert option_begin_state is not None
                time_difference = time_steps_under_option
                discounted_future_value = (self.gamma ** time_difference) * np.max(q[new_state, :])
                old_q = Q[option_begin_state, action_idx]
                delta_Q = discounted_reward_under_option + discounted_future_value - old_q

                Q[old_state][primitive_action] += self.alpha * delta_Q

                time_steps_under_option = 0
                option_begin_state = None

            elif self.is_option(action_idx):
                # following option
                time_steps_under_option += 1
                # register the reward following option
                discounted_reward_under_option += reward * (self.gamma ** time_steps_under_option)

                # update the q-value for the last transition off-policy
                delta_Q = reward + self.gamma * np.max(q[new_state, :]) - Q[old_state, action_idx]
                Q[old_state][action_idx] += self.alpha * delta_Q

            else:
                delta_Q = reward + self.gamma * np.max(q[new_state, :]) - Q[old_state, action_idx]
                Q[old_state][action_idx] += self.alpha * delta_Q

            # if False and self.test_run and len(self.available_actions) < 11:  # and time_steps_under_option == 1:
            #     render = 2
            #     if self.is_option(action):
            #         if primitive_action == -1:
            #             nice_act = -1
            #         else:
            #             nice_act = envs.boxes.BoxWorldActions(primitive_action)

            #         target = np.argwhere(action == -1)[0]
            #         print(step, "pos", self.environment.agent_position_idx, "act", target, nice_act)
            #         highlight_square = target
            #     else:
            #         if isinstance(self.environment, envs.boxes.BoxWorld):
            #             a = envs.boxes.BoxWorldActions(action)
            #         elif isinstance(self.environment, envs.hungry_thirsty.HungryThirsty):
            #             a = envs.hungry_thirsty.HungryThirstyActions(action)
            #         else:
            #             raise NotImplementedError()
            #         print(step, "pos", self.environment.agent_position_idx, "act", a, self.Q1[old_state, action_idx])
            #         highlight_square = new_state
            # else:
            #    highlight_square = None

            # if abs(delta_Q) < 0.001:
            #     no_change += 1
            # else:
            #     no_change = 0

            if generate_options and delta_Q > 1 and self.environment.agent_position_idx not in option_goals:
                option_goals.add(self.generate_option())

            # if steps_of_no_change is not None and no_change > max_no_change:
            #     max_no_change = no_change
            #     if plot_progress and self.learning_option:
            #         progress_bar.update(1)

            old_state = new_state
            #

            # if self.test_run and reward > 0:
            #     render = self.render_board(render, highlight_square, sleep_time=1)
            # elif self.is_option(action) or primitive_action < 4:
            #     render = self.render_board(render, highlight_square, sleep_time=0)
            # else:
            #     try:
            #         relevant_positions = (self.environment.water_position, self.environment.food_position)
            #     except AttributeError:
            #         relevant_positions = self.environment._state['box'].keys()

            #     if self.environment.agent_position_idx in relevant_positions:
            #        render = self.render_board(render, highlight_square, sleep_time=0)

            # if plot_progress and (self.train_run or self.test_run):
            #     progress_bar.update(1)

            # if steps_of_no_change is not None and no_change > steps_of_no_change:
            #    break

        opts = self.available_actions[self.environment.action_space.n:]
        return opts, cumulative_reward

    def render_board(self, render, highlight_square=None, sleep_time=1. / 30.):
        if render > 0:
            render -= 1
            time.sleep(sleep_time)
            self.environment.show_board(
                some_matrix=np.sum(self.Q1 + self.Q2, axis=1),
                policy=np.argmax(self.Q1 + self.Q2, axis=1),
                highlight_square=highlight_square,
            )
            if render == 0:
                input("sleeping")
        return render

    def generate_option(self):
        goal = self.environment.agent_position_idx
        new_option = learn_option(goal, self.environment)

        # TODO: REMOVE HACK
        if new_option.shape[0] < self.environment.observation_space.n:
            # TODO: remove print("OPTION SIZE MISMATCH, TILING")
            new_option = np.tile(
                new_option[:self.environment.number_of_tiles],
                self.environment.observation_space.n // self.environment.number_of_tiles
            )

        new_option = tuple(new_option)
        # new_option = learn_option(old_state, self.environment)
        self.available_actions.append(new_option)
        option_idx = self.Q1.shape[1] + 1
        self.action_to_id[new_option] = option_idx - 1
        tmp_Q = np.empty((self.Q1.shape[0], option_idx))
        tmp_Q[:, :-1] = self.Q1
        self.Q1 = tmp_Q
        self.Q1[:, -1] = self.Q1[:, :-1].mean(axis=1)
        tmp_Q = np.empty((self.Q2.shape[0], option_idx))
        tmp_Q[:, :-1] = self.Q2
        self.Q2 = tmp_Q
        self.Q2[:, -1] = self.Q2[:, :-1].mean(axis=1)
        return self.environment.agent_position_idx

    def test(self, eval_steps):
        cumulative_reward = 0
        primitive_action_idx = None
        action_idx = None

        for tile_idx in range(self.environment.number_of_tiles):
            old_state = self.environment.reset()
            self.environment.teleport_agent(tile_idx)

            for step in range(eval_steps):
                action_idx, primitive_action_idx = self.pick_action(old_state,
                                                                    old_primitive_action=primitive_action_idx,
                                                                    old_action_idx=action_idx)

                if primitive_action_idx == -1:
                    continue

                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action_idx])
                cumulative_reward += reward

                # self.render_board(render=2)

                old_state = new_state
        return cumulative_reward / self.environment.number_of_tiles

    def is_option(self, action):
        # return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)
        return action is not None and action >= self.nr_primitive_actions


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


@disk_utils.disk_cache
def learn_option(goal, mdp):
    print("\ngenerating policy for goal:{}\n".format(goal))

    def surrogate_reward(_mdp):
        # return 1 if goal == _mdp._hash_state() else -1
        return 1 if goal == _mdp.agent_position_idx else -1

    # learner = learners.policy_iter.PolicyIteration(
    #     env=mdp,
    #     options=None,
    #     epsilon=0.1,
    #     gamma=0.90,
    #     alpha=0.1,
    #     surrogate_reward=surrogate_reward,
    # )
    # # TODO: re-enable QLearning this pol iter is for deterministic envs
    # value, option = learner.solvePolicyIteration()

    simple_mdp = envs.gridworld.GridWorld(side_size=6, terminal_states=(), start_from_borders=True)
    learner = QLearning(
        env=simple_mdp,
        options=None, epsilon=0.1, gamma=0.90, alpha=0.1, surrogate_reward=surrogate_reward,
        learning_option=True
    )
    _ = learner.learn(max_steps=1000000)
    option = np.argmax(learner.Q1 + learner.Q2, axis=1)

    state_idx = goal
    try:
        while True:
            option[state_idx] = -1
            state_idx += mdp.number_of_tiles
    except IndexError as e:
        pass
    simple_mdp.print_board(
        some_matrix=np.max(learner.Q1 + learner.Q2, axis=1),
        # some_matrix=value,
        policy=option,
    )
    input("done")
    option = np.tile(
        option, mdp.observation_space.n // mdp.number_of_tiles
    )
    # time.sleep(1000)
    return option
