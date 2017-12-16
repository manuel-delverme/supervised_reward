import numpy as np
import heapq
import tqdm
import random


class DoubleQLearning(object):
    def __init__(self, env, surrogate_reward=None):
        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        for state in env.terminal_states:
            self.Q1[state, :] = 0
            self.Q2[state, :] = 0
        self.environment = env
        self.surrogate_reward = surrogate_reward

    @staticmethod
    def null_criterion(*args, **kwargs):
        return False

    def learn(self, alpha, gamma, epsilon, training_episodes_nr, num_goals=10, goal_criterion=null_criterion):
        # returnSum = 0.0
        print("[double_q] learn")
        goals = []
        goal_states = set()
        for episode_num in tqdm.tqdm(range(training_episodes_nr)):
            old_state = self.environment.reset()
            terminal = False
            # if self.surrogate_reward is not None:
            #     reward = self.surrogate_reward(self.environment)

            while not terminal:
                if epsilon > random.random():
                    qTotal = self.Q1 + self.Q2
                    action = np.argmax(qTotal[old_state, :])
                else:
                    action = random.randint(0, self.environment.action_space.n - 1)

                new_state, reward, terminal, info = self.environment.step(action)
                if self.surrogate_reward is not None:
                    reward = self.surrogate_reward(self.environment)

                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1
                best_future_act = np.argmax(Q[new_state, :])
                old_Q = Q[old_state, action]
                delta_Q = alpha * (reward + (gamma * q[new_state, best_future_act]) - old_Q)
                old_Q += delta_Q

                # if goal_criterion(old_state, delta_Q):
                #     goals.append(old_state)
                # TODO: or new state?
                if len(goals) < num_goals and new_state not in goal_states:
                    goal_states.add(new_state)
                    heapq.heappush(goals, (delta_Q, new_state))
                elif delta_Q > goals[0][0]:
                    if new_state not in goal_states:
                        # print("replacing", goals[0], "with", (delta_Q, new_state))
                        goal_states.add(new_state)
                        removed_item = heapq.heapreplace(goals, (delta_Q, new_state))
                        goal_states.remove(removed_item[1])
                    else:
                        for goal_idx, (goal_dq, state_idx) in enumerate(goals):
                            if state_idx == new_state and goal_dq < delta_Q:
                                # print("updating", state_idx, "with", (delta_Q, new_state))
                                goals[goal_idx] = (delta_Q, state_idx)
                                heapq.heapify(goals)

                old_state = new_state
        return goals


def qLearningWithOptions(env, alpha, gamma, epsilon, maxLengthEp, nEpisodes, loadedOptions=None):
    options = loadedOptions
    actionSetPerOption = []

    for i in range(len(loadedOptions)):
        tempActionSet = list(range(env.action_space.n))
        tempActionSet.append('terminate')
        actionSetPerOption.append(tempActionSet)

    returns_eval = []
    returns_learn = []
    # option_set_size = 4
    returns_eval.append([])
    returns_learn.append([])

    # print('Using', option_set_size, 'options at time')

    actionSet = list(range(env.action_space.n)) + options
    learner = QLearning(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        environment=env,
        seed=1,
        useOnlyPrimActions=False,
        actionSet=actionSet,
        actionSetPerOption=actionSetPerOption
    )
    # learner = DoubleQLearning(environment=env, options)

    cum_reward = 0
    # returns_learn = np.zeros(len(options), nEpisodes)
    for episode_idx in tqdm.tqdm(range(nEpisodes)):
        _ = learner.learnOneEpisode()
        cum_reward += learner.evaluateOneEpisode()
    return cum_reward


class QLearning:
    Q = None
    environment = None
    alpha = None
    gamma = 0.9
    epsilon = 0.05
    numStates = 0
    actionSet = None
    optionsActionSet = None
    numPrimitiveActions = -1
    toLearnUsingOnlyPrimitiveActions = False

    def __init__(self, alpha, gamma, epsilon, environment, seed=1, useOnlyPrimActions=False, actionSet=None,
                 actionSetPerOption=None):

        """Initialize variables that are useful everywhere."""
        print("Qlearning using options", not useOnlyPrimActions)
        self.environment = environment
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.numStates = self.environment.observation_space.n
        self.numPrimitiveActions = self.environment.action_space.n
        self.toLearnUsingOnlyPrimitiveActions = useOnlyPrimActions
        random.seed(seed)

        self.actionSet = actionSet
        self.optionsActionSet = actionSetPerOption

        if self.toLearnUsingOnlyPrimitiveActions:
            if self.epsilon != 1.0:
                print('Something will go wrong. Epsilon should be 1.0 when using the options only for exploration in '
                      'QLearning.')

            self.Q = np.zeros((self.numStates, self.numPrimitiveActions))
        else:
            self.Q = np.zeros((self.numStates, len(self.actionSet)))

    def getAvailableActionSet(self, s):
        availActions = []
        for i in range(len(self.actionSet)):
            if i < self.numPrimitiveActions:
                availActions.append(i)
            elif self.getPrimitiveAction(s, i) != 'terminate':
                availActions.append(i)

        return availActions

    def getIdFromPrimitiveActions(self, action):
        for i in range(self.numPrimitiveActions):
            if self.environment.getActionSet()[i] == action:
                return i

        return 'error'

    def epsilonGreedy(self, F, s, epsilon=None):
        """ Epsilon-greedy function. F needs to be Q[s], so it
			consists of one value per action."""
        if epsilon is None:
            epsilon = self.epsilon
        rnd = random.uniform(0, 1)

        availActions = self.getAvailableActionSet(s)

        if rnd <= epsilon:
            idx = random.randrange(0, len(availActions))
            return availActions[idx]
        else:
            if self.toLearnUsingOnlyPrimitiveActions:
                availActions = list(range(len(self.environment.getActionSet())))

            T = F[availActions]
            idx = np.random.choice(np.where(T == T.max())[0])
            return availActions[idx]

    def getPrimitiveAction(self, s, a):
        if a < self.numPrimitiveActions:
            return self.actionSet[a]
        else:
            idxOption = a - self.numPrimitiveActions
            return self.optionsActionSet[idxOption][self.actionSet[a][s]]

    def learnOneEpisode(self, timestepLimit=1000):
        """Execute Q-learning for one episode."""

        timestep = 0
        previous_action = -1
        cumulativeReward = 0

        old_state = self.environment.reset()
        terminal = False
        while not terminal and timestep < timestepLimit:
            if previous_action < self.numPrimitiveActions:
                action = self.epsilonGreedy(self.Q[old_state], old_state)
            action = self.getPrimitiveAction(old_state, action)

            if action == 'terminate':
                action = self.epsilonGreedy(self.Q[old_state], old_state)
                action = self.getPrimitiveAction(old_state, action)

            previous_action = action
            new_state, reward, terminal, info = self.environment.step(action)
            # reward = self.environment.step(action)
            cumulativeReward += reward
            # sNext = self.environment.getCurrentState()

            if self.toLearnUsingOnlyPrimitiveActions:
                action = self.getIdFromPrimitiveActions(action)
            self.Q[old_state][action] = self.Q[old_state][action] + self.alpha * (
                reward + self.gamma * np.max(self.Q[new_state]) - self.Q[old_state][action])

            old_state = new_state
            timestep += 1

        return cumulativeReward

    def evaluateOneEpisode(self, eps=None, timestepLimit=1000):
        """Evaluate Q-learning for one episode."""

        reward = 0
        timestep = 0
        previous_action = -1
        cumulativeReward = 0

        old_state = self.environment.reset()
        terminal = False

        while not terminal and timestep < timestepLimit:
            if previous_action < self.numPrimitiveActions:
                action = self.epsilonGreedy(self.Q[old_state], old_state, epsilon=eps)
            action = self.getPrimitiveAction(old_state, action)

            if action == 'terminate':
                action = self.epsilonGreedy(self.Q[old_state], old_state, epsilon=eps)
                action = self.getPrimitiveAction(old_state, action)

            previous_action = action
            new_state, reward, terminal, info = self.environment.step(action)
            # reward = self.environment.step(action)
            cumulativeReward += reward
            # sNext = self.environment.getCurrentState()

            old_state = new_state
            timestep += 1
        return cumulativeReward
