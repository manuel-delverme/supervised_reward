import sys
from environment import GridWorld
from learning import Learning
from qlearning import QLearning

import envs.gridworld


def policyEvaluation(env):
    ''' Simple test for policy evaluation '''

    pi = numStates * [[0.25, 0.25, 0.25, 0.25]]
    actionSet = env.getActionSet()

    # This solution is slower and it does not work for gamma = 1
    # polEval = Learning(0.9999, env, augmentActionSet=False)
    # expectation = polEval.solvePolicyEvaluation(pi)

    bellman = Learning(1, env, augmentActionSet=False)
    expectation = bellman.solveBellmanEquations(pi, actionSet, None)

    for i in range(len(expectation) - 1):
        sys.stdout.write(str(expectation[i]) + '\t')
        if (i + 1) % env.numCols == 0:
            print()
    print()


def policyIteration(env):
    ''' Simple test for policy iteration '''

    polIter = Learning(0.9, env, augmentActionSet=False)
    V, pi = polIter.solvePolicyIteration()

    # I'll assign the goal as the termination action
    pi[env.getGoalState()] = 4

    # Now we just plot the learned value function and the obtained policy
    plot = Plotter(outputPath, env)
    plot.plotValueFunction(V[0:numStates], 'goal_')
    plot.plotPolicy(pi[0:numStates], 'goal_')


def qLearningWithOptions(env, alpha, gamma, options_eps, epsilon, maxLengthEp, nEpisodes, useNegation,
                         loadedOptions=None):
    options = loadedOptions
    actionSetPerOption = []

    for i in range(len(loadedOptions)):
        tempActionSet = env.getActionSet()
        tempActionSet.append('terminate')
        actionSetPerOption.append(tempActionSet)

    returns_eval = []
    returns_learn = []
    option_set_size = 4
    returns_eval.append([])
    returns_learn.append([])

    print('Using', option_set_size, 'options at time')

    actionSet = env.getActionSet() + options[:option_set_size]
    num_primitives = len(actionSet) - option_set_size

    learner = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, environment=env, seed=1, useOnlyPrimActions=False,
                        actionSet=actionSet, actionSetPerOption=actionSetPerOption)

    # returns_learn = np.zeros(len(options), nEpisodes)
    candidate_options = [i for i in range(option_set_size)]
    for incumbent_idx in range(option_set_size, len(options)):
        for episode_idx in range(nEpisodes):
            # returns_learn[incumbent_idx - numOptionsToUse][episode_idx] = learner.learnOneEpisode(timestepLimit=maxLengthEp)
            # returns_eval[incumbent_idx - numOptionsToUse][episode_idx] = learner.evaluateOneEpisode(eps=0.01, timestepLimit=maxLengthEp)
            _ = learner.learnOneEpisode(timestepLimit=maxLengthEp)
            _ = learner.evaluateOneEpisode(eps=0.01, timestepLimit=maxLengthEp)

        performances = learner.Q.sum(axis=0)
        opt_performance = performances[num_primitives:]
        primitive_performance = performances[:num_primitives]
        print("performance: primitive-", primitive_performance, "option-", opt_performance)
        performers = opt_performance.argsort()
        worst_performer_idx = performers[0]
        # TODO: think
        #
        #  learner.Q[:, num_primitives + worst_performer_idx] = 0.5 * learner.Q.mean(axis=1)

        # hype driven exploration
        # learner.Q[:, num_primitives + worst_performer_idx] = learner.Q.max(axis=1).copy()

        # lobotomy
        learner.Q.fill(0)

        learner.actionSet[num_primitives + worst_performer_idx] = options[incumbent_idx]

        worst_performer = candidate_options[worst_performer_idx]
        candidate_options[worst_performer_idx] = incumbent_idx
        print("replaced {}(score {})".format(worst_performer, opt_performance[worst_performer_idx]))
    print("best options found", sorted(candidate_options))
    return [options[idx] for idx in candidate_options]


class Regressor(object):
    def get_reward_function(self):
        pass


class GoalSelector(object):
    def select_goals(self, mdp, reward_function):
        pass


class EnvGenerator(object):
    def __init__(self, _):
        pass


if __name__ == "__main__":
    # env = envs.gridworld.GridWorld()
    mdp_distribution = EnvGenerator(envs.gridworld.GridWorld)
    regressor = Regressor()
    goal_selector = GoalSelector()
    TRAINING_SIZE = 10
    TEST_SIZE = 3

    reward_function = regressor.get_reward_function()
    for idx, mdp in enumerate(mdp_distribution):
        if idx > TRAINING_SIZE:
            break
        goals = goal_selector.select_goals(mdp, reward_function)
        options = []
        for goal in goals:
            mdp.set_goal(goal)
            pol_iter = policyIteration(env=mdp)
            v, pi = pol_iter.solvePolicyIteration()
            options.append(pi)

    cum_cum_reward = 0
    for idx, mdp in enumerate(mdp_distribution):
        if idx > TEST_SIZE:
            break
        cum_reward = qLearningWithOptions(env=mdp)
    print(cum_cum_reward / TEST_SIZE)

    # terminal = False
    # while not terminal:
    #     state, reward, terminal, info = env.step(1)
    #     env.render(mode="ansi")
