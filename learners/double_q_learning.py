import numpy as np
import envs.gridworld
import collections
import tqdm
import learners.policy_iter
import sys
import time
import disk_utils
import envs.boxes
import envs.hungry_thirsty
import random
import sys


class QLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.3, surrogate_reward=None, test_run=False,
                 learning_option=False, train_run=False):

        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.previous_action = None
        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.available_actions = list(range(self.environment.action_space.n))

        assert (train_run ^ test_run ^ learning_option)
        self.test_run = test_run
        self.train_run = train_run
        self.learning_option = learning_option

        action_size = env.action_space.n
        self.action_to_id = {k: k for k in range(action_size)}
        if options is not None:
            for idx in range(len(options)):
                options[idx].flags.writeable = False

            self.available_actions.extend(options)
            action_size += len(options)
            for option in options:
                # goal = np.argwhere(action == -1)[0]
                self.action_to_id[option.data.tobytes()] = len(self.action_to_id)

        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, action_size)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, action_size)

    def pick_action(self, old_state, old_action, kill_option=False):
        TERMINATE_OPTION = -1
        if old_action != TERMINATE_OPTION:
            if kill_option:
                return self.previous_action, TERMINATE_OPTION, self.action_to_id[self.previous_action.data.tobytes()]

            # i was following an option and should still follow it
            if is_option(self.previous_action):
                # keep going
                # return self.previous_action, self.previous_action[old_state]
                return self.previous_action, self.previous_action[old_state], self.action_to_id[
                    self.previous_action.data.tobytes()]

        # if the option terminated OR i was using primitives
        # Q_exploit
        primitive_action = TERMINATE_OPTION
        # epsilon could be silly and choose a terminated option
        # pick again is faster than checking for initiation sets
        while primitive_action == TERMINATE_OPTION:
            if self.epsilon < random.random():
                best_action = np.argmax(self.Q1[old_state] + self.Q2[old_state])
                action = self.available_actions[best_action]
            else:
                action = random.choice(self.available_actions)

            if is_option(action):
                primitive_action = action[old_state]
            else:
                primitive_action = action

        if is_option(action):
            action_id = self.action_to_id[action.data.tobytes()]
        else:
            action_id = self.action_to_id[action]
        return action, primitive_action, action_id

    def learn(self, steps_of_no_change=None, generate_options=False, max_steps=None, plot_progress=False):
        assert (steps_of_no_change is not None or max_steps is not None)

        if plot_progress:
            if steps_of_no_change is not None:
                progress_bar = tqdm.tqdm(total=steps_of_no_change, file=sys.stdout)
            elif self.train_run or self.test_run:
                progress_bar = tqdm.tqdm(total=max_steps, file=sys.stdout)

        cumulative_reward = 0
        terminal = True
        time_steps_under_option = 0
        discounted_reward_under_option = 0
        no_change = 0
        max_no_change = 0
        render = 0
        option_goals = set()
        old_states = collections.deque(maxlen=20)
        old_actions = collections.deque(maxlen=20)
        primitive_action = None
        option_begin_state = None
        stepss_to_goal = []
        steps_to_goal = 0

        for step in range(max_steps):
            # if step % (max_steps/10) == 0:
            #     render = 50
            if terminal:
                old_state = self.environment.reset()
                stepss_to_goal.append(steps_to_goal)
                steps_to_goal = 0

            if len(old_states) == 20 and len(set(old_states)) < 4 and is_option(
                    action) and time_steps_under_option >= 19:
                kill_option = True
                print("killed at:", time_steps_under_option, "\n", np.argwhere(action == -1))
                print(old_states, "\n", old_actions)
                self.environment.render_board(policy=action)
                input()
                old_states.clear()
            else:
                kill_option = False

            action, primitive_action, action_idx = self.pick_action(
                old_state, old_action=primitive_action, kill_option=kill_option
            )
            if option_begin_state is None and is_option(action):
                option_begin_state = old_state

            old_states.append(old_state)
            old_actions.append(primitive_action)

            # if old_pick == primitive_action:
            #     if is_option(action) and self.test_run:
            #         print("picked option", time_steps_under_option, envs.boxes.BoxWorldActions(primitive_action))
            #         self.environment.print_board(policy=action)
            #         print("uffa")
            # else:
            #     old_pick = primitive_action

            if primitive_action != -1:
                new_state, reward, terminal, info = self.environment.step(primitive_action)
                steps_to_goal += 1

                if self.surrogate_reward is not None:
                    reward = self.surrogate_reward(self.environment)
                    if reward > 0 and self.learning_option:
                        terminal = True
                    if reward > 0 and self.training_run:
                        terminal = True

                cumulative_reward += reward
                # cumulative_rewards.append(cumulative_reward)

            if False and self.test_run and len(self.available_actions) < 11:  # and time_steps_under_option == 1:
                render = 2
                if is_option(action):
                    if primitive_action == -1:
                        nice_act = -1
                    else:
                        nice_act = envs.boxes.BoxWorldActions(primitive_action)

                    target = np.argwhere(action == -1)[0]
                    print(step, "pos", self.environment.agent_position_idx, "act", target, nice_act)
                    highlight_square = target
                else:
                    if isinstance(self.environment, envs.boxes.BoxWorld):
                        a = envs.boxes.BoxWorldActions(action)
                    elif isinstance(self.environment, envs.hungry_thirsty.HungryThirsty):
                        a = envs.hungry_thirsty.HungryThirstyActions(action)
                    else:
                        raise NotImplementedError()
                    print(step, "pos", self.environment.agent_position_idx, "act", a, self.Q1[old_state, action_idx])
                    highlight_square = new_state
            else:
                highlight_square = None

            if is_option(action) and primitive_action != -1:
                time_steps_under_option += 1
                discounted_reward_under_option += reward * (self.gamma ** time_steps_under_option)

            Q, q = (self.Q1, self.Q2) if random.random() > 0.5 else (self.Q2, self.Q1)

            if primitive_action == -1:
                assert option_begin_state is not None
                time_difference = time_steps_under_option
                time_steps_under_option = 0
                discounted_future_value = (self.gamma ** time_difference) * np.max(q[new_state, :])
                old_q = Q[option_begin_state, action_idx]
                delta_Q = discounted_reward_under_option + discounted_future_value - old_q

                Q[old_state][action_idx] += self.alpha * delta_Q
            else:
                delta_Q = reward + self.gamma * np.max(q[new_state, :]) - Q[old_state, action_idx]
                Q[old_state][action_idx] += self.alpha * delta_Q

            if abs(delta_Q) < 0.001:
                no_change += 1
            else:
                no_change = 0

            if generate_options and delta_Q > 1 and self.environment.agent_position_idx not in option_goals:
                option_goals.add(self.generate_option())
                # for cheat_pos in {0, 5, 30, 35}.difference(option_goals):
                #     self.environment.teleport_agent(cheat_pos)
                #     option_goals.add(self.generate_option())
                #     break
                #     # else:
                #     #     option_goals.add(self.generate_option())

            if steps_of_no_change is not None and no_change > max_no_change:
                max_no_change = no_change
                if plot_progress and self.learning_option:
                    progress_bar.update(1)

            old_state = new_state
            self.previous_action = action

            if False and self.test_run and reward > 0:
                render = self.render_board(render, highlight_square, sleep_time=1)
            elif is_option(action) or primitive_action < 4:
                render = self.render_board(render, highlight_square, sleep_time=0)
            else:
                try:
                    relevant_positions = (self.environment.water_position, self.environment.food_position)
                except AttributeError:
                    relevant_positions = self.environment._state['box'].keys()

                if self.environment.agent_position_idx in relevant_positions:
                    render = self.render_board(render, highlight_square, sleep_time=0)

            if plot_progress and (self.train_run or self.test_run):
                progress_bar.update(1)

            if steps_of_no_change is not None and no_change > steps_of_no_change:
                break

            if primitive_action == -1:
                option_begin_state = None
        else:
            pass
            # print("max step, break")
        opts = self.available_actions[self.environment.action_space.n:]
        # if len(self.available_actions[self.environment.action_space.n:]) > 0:
        #     el = random.randint(1, len(self.available_actions[self.environment.action_space.n:]))
        #     el = 1
        #     if el > 0:
        #         opts = random.sample(opts, el)
        return opts, stepss_to_goal[1:]

    def render_board(self, render, highlight_square=None, sleep_time=1./30.):
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

        new_option.flags.writeable = False
        # new_option = learn_option(old_state, self.environment)
        self.available_actions.append(new_option)
        option_idx = self.Q1.shape[1] + 1
        self.action_to_id[new_option.data.tobytes()] = option_idx - 1
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
        terminal = False
        primitive_action = None
        cumulative_reward = 0

        for tile_idx in range(self.environment.number_of_tiles):
            old_state = self.environment.reset()
            self.environment.teleport_agent(tile_idx)

            for step in range(eval_steps):
                action, primitive_action, action_idx = self.pick_action(old_state, old_action=primitive_action)
                if primitive_action == -1:
                    continue

                new_state, reward, terminal, info = self.environment.step(primitive_action)
                cumulative_reward += reward

                # self.render_board(render=2)

                old_state = new_state
                self.previous_action = action
        return cumulative_reward / self.environment.number_of_tiles


def is_option(action):
    # return isinstance(action, np.ndarray)
    return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)


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
