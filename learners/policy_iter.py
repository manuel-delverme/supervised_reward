''' Author: Marlos C. Machado '''
import math
import numpy as np
import random


class Learning(object):
    V = None
    pi = None
    gamma = 0.9
    numStates = 0
    actionSet = None
    environment = None

    def __init__(self, gamma, env, surrogate_reward, augmentActionSet=False):
        self.gamma = gamma
        self.environment = env
        self.numStates = env.observation_space.n
        self.surrogate_reward = surrogate_reward

        self.V = np.zeros(self.numStates)
        self.pi = np.zeros(self.numStates, dtype=np.int)

        if augmentActionSet:
            self.actionSet = np.append(env.getActionSet(), ['terminate'])
        else:
            self.actionSet = list(range(env.action_space.n))

    def _evalPolicy(self):
        delta = 0.0
        for s in range(self.numStates):
            v = self.V[s]
            action = self.actionSet[self.pi[s]]

            self.environment.teleport_agent(s)
            nextState, nextReward, terminal, info = self.environment.step(action)

            # if nextState == 4:
            #   print(s, ":", self.V[s], "becomes", end=" ")

            if self.surrogate_reward is not None:
                nextReward = self.surrogate_reward(self.environment)

            self.V[s] = nextReward + self.gamma * self.V[nextState]
            # if nextState == 4:
            #     print(self.V[s])
            delta = max(delta, math.fabs(v - self.V[s]))

        return delta

    def _improvePolicy(self):
        """ Policy improvement step. """
        policy_stable = True
        for s in range(self.numStates):
            old_action = self.pi[s]
            tempV = [0.0] * len(self.actionSet)
            # I first get all value-function estimates
            for i in range(len(self.actionSet)):

                self.environment.teleport_agent(s)
                nextState, nextReward, terminal, info = self.environment.step(self.actionSet[i])
                if self.surrogate_reward is not None:
                    nextReward = self.surrogate_reward(self.environment)

                tempV[i] = nextReward + self.gamma * self.V[nextState]

            # Now I take the argmax
            self.pi[s] = np.argmax(tempV)
            # I break ties always choosing to terminate:
            if math.fabs(tempV[self.pi[s]] - tempV[(len(self.actionSet) - 1)]) < 0.001:
                self.pi[s] = (len(self.actionSet) - 1)
            if old_action != self.pi[s]:
                policy_stable = False

        return policy_stable

    def solvePolicyIteration(self, theta=0.001):
        """ Implementation of Policy Iteration, as in the policy iteration pseudo-code presented in Sutton and Barto
        (2016). """

        policy_stable = False
        while not policy_stable:
            # Policy evaluation
            delta = self._evalPolicy()
            while theta < delta:
                delta = self._evalPolicy()

            # Policy improvement
            policy_stable = self._improvePolicy()
        return self.V, self.pi

    def solvePolicyEvaluation(self, pi, theta=0.001):
        '''Implementation of Policy Evaluation, as in the policy evaluation
		   pseudo-code presented in Sutton and Barto (2016).'''

        # I'll use the same V, it shouldn't really matter,
        # although ideally these things should be independent
        self.V = np.zeros(self.numStates + 1)
        iteration = 1

        delta = 1
        while delta > theta:
            delta = 0
            for s in range(self.numStates - 1):
                v = self.V[s]
                tempSum = 0
                for a in range(len(pi[s])):
                    nextS, nextR = self.environment.getNextStateAndReward(
                        s, self.actionSet[a])
                    tempSum += pi[s][a] * 1.0 * (
                        nextR + self.gamma * self.V[nextS])

                self.V[s] = tempSum
                delta = max(delta, math.fabs(v - self.V[s]))

            if iteration % 1000 == 0:
                print('Iteration:', iteration, '\tDelta:', delta)
            iteration += 1

        '''
		import sys
		for i in xrange(16):
			sys.stdout.write(str(self.V[i]) + ' ')
			if (i + 1) % 4 == 0:
				print
		'''
        return self.V

    def solveBellmanEquations(self, pi, fullActionSet, optionsActionSet):
        ''' This method generates the Bellman equations using the model
			available in self.environment and solves the generated set of
			linear equations.'''

        numberOfPrimitiveActions = 4
        # ax = b
        a_equations = np.zeros((self.numStates, self.numStates))
        b_equations = np.zeros(self.numStates)

        '''
		# V[s] = \sum \pi(a|s) \sum p(s',r|s,a) [r + \gamma V[s']]
		# V[s] = \sum \pi(a|s) 1.0 [r + \gamma V[s']] (assuming determinism)
		# - \sum \pi(a|s) r = -V[s] + \sum \pi(a|s) \gamma V[s']
		'''
        for s in range(self.numStates - 1):
            a_equations[s][s] = -1
            for a in range(len(pi[s])):
                nextS = -1
                nextR = -1

                # If it is a primitive action
                if isinstance(fullActionSet[a], str):
                    nextS, nextR = self.environment.getNextStateAndReward(
                        s, fullActionSet[a])
                else:  # if it is an option
                    nextS, nextR = self.environment.getNextStateAndRewardFromOption(
                        s, fullActionSet[a],
                        optionsActionSet[a - numberOfPrimitiveActions])

                a_equations[s][nextS] += pi[s][a] * self.gamma
                b_equations[s] -= pi[s][a] * nextR

        for i in range(self.numStates):
            hasOnlyZeros = True
            for j in range(self.numStates):
                if a_equations[i][j] != 0.0:
                    hasOnlyZeros = False

            if hasOnlyZeros:
                a_equations[i][i] = 1
                b_equations[i] = 0

        expectation = np.linalg.solve(a_equations, b_equations)
        return expectation

def discoverOptions(env, epsilon, discoverNegation):
    # I'll need this when computing the expected number of steps:
    options = []
    actionSetPerOption = []

    # Computing the Combinatorial Laplacian
    W = env.getAdjacencyMatrix()
    numStates = env.getNumStates()
    D = np.zeros((numStates, numStates))

    # Obtaining the Valency Matrix
    diag = np.sum(W, axis=0)

    # Making sure our final matrix will be full rank
    diag = np.clip(diag, 1.0, np.inf)
    D2 = np.diag(diag)

    # Normalized Laplacian
    L = D - W
    D2[D2 != 0] = np.power(D2[D2 != 0], -0.5)
    expD = D2
    normalizedL = expD.dot(L).dot(expD)

    # Eigendecomposition
    # IMPORTANT: The eigenvectors are in columns
    eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
    # I need to sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # If I decide to use both directions of the eigenvector, I do it here.
    # It is easier to just change the list eigenvector, even though it may
    # not be the most efficient solution. The rest of the code remains the same.
    if discoverNegation:
        oldEigenvalues = eigenvalues
        oldEigenvectors = eigenvectors.T
        eigenvalues = []
        eigenvectors = []
        for i in range(len(oldEigenvectors)):
            eigenvalues.append(oldEigenvalues[i])
            eigenvalues.append(oldEigenvalues[i])
            eigenvectors.append(oldEigenvectors[i])
            eigenvectors.append(-1 * oldEigenvectors[i])

        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors).T

    # if plotGraphs:
    #     # Plotting all the basis
    #     plot = Plotter(outputPath, env)
    #     plot.plotBasisFunctions(eigenvalues, eigenvectors)

    # Now I will define a reward function and solve the MDP for it
    # I iterate over the columns, not rows. I can index by 0 here.
    print("Solving for eigenvector #", end=' ')
    vecs = eigenvectors.T.copy()
    np.random.shuffle(vecs)

    for idx, eigenvector in enumerate(vecs):
        print(idx, end=' ')
        polIter = Learning(0.9, env, augmentActionSet=True)
        env.defineRewardFunction(eigenvector)
        V, pi = polIter.solvePolicyIteration(surrogate_reward=surrogate_reward)

        # Now I will eliminate any actions that may give us a small improvement.
        # This is where the epsilon parameter is important. If it is not set all
        # it will never be considered, since I set it to a very small value
        for j in range(len(V)):
            if V[j] < epsilon:
                pi[j] = len(env.getActionSet())

        options.append(pi[0:numStates])
        optionsActionSet = env.getActionSet()
        optionsActionSet.append('terminate')
        actionSetPerOption.append(optionsActionSet)

    print("\n")
    # I need to do this after I'm done with the PVFs:
    env.defineRewardFunction(None)
    env.reset()
    return options, actionSetPerOption


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


def policyIteration(env, surrogate_reward=None):
    polIter = Learning(0.9, env, augmentActionSet=False, surrogate_reward=surrogate_reward)
    V, pi = polIter.solvePolicyIteration()

    # I'll assign the goal as the termination action
    for state in env.terminal_states:
        pi[state] = -1

    # Now we just plot the learned value function and the obtained policy
    env.plot_policy_and_value(pi, V)
    return pi

