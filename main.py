import multiprocessing

import numpy as np
import tqdm

import config
import envs.boxes
import envs.hungry_thirsty
import envs.minigrid
import shared.constants as C
import shared.utils
from fitness import fitness_function


class Reward:
    def __init__(self, reward_vector):
        self.reward_vector = reward_vector

    def __call__(self, observation, environment):
        image = observation.reshape(-1)
        return self.reward_vector.dot(image)

    def __str__(self):
        return str(self.reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size))

    def __repr__(self):
        return repr(self.reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size))

    def reset(self):
        pass

    def motivating_function(self, state):
        class MotivatingFunction(Reward):
            def __repr__(self):
                representation = self.reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size)
                # for layer in range(representation.shape[0]):
                #     representation[layer]
                # return repr()
                return repr(representation)

        motivating_function = np.multiply(self.reward_vector, state.reshape(-1))
        negative_rewards = self.reward_vector[self.reward_vector < 0]
        # keep the punishments
        motivating_function[self.reward_vector < 0] = negative_rewards

        return MotivatingFunction(motivating_function)


class ConstrainedReward(Reward):
    def __init__(self, reward_parameters, normalize=True):
        self.reward_parameters = reward_parameters
        self.reward_coords = []
        points = reward_parameters.reshape(-1, 3)
        reward = np.ones(shape=(config.Minigrid.nr_layers, config.agent_view_size, config.agent_view_size)) * -0.005
        for layer, x, y in points:
            if normalize:
                x *= config.agent_view_size / 2
                x += config.agent_view_size / 2
                x = np.clip(x, 0, config.agent_view_size - 1)

                y *= config.agent_view_size / 2
                y += config.agent_view_size / 2
                y = np.clip(y, 0, config.agent_view_size - 1)

                layer *= config.Minigrid.nr_layers
                layer += config.Minigrid.nr_layers / 2
                layer = np.clip(layer, 0, config.Minigrid.nr_layers - 1)
            reward[int(layer), int(x), int(y)] = 1
            self.reward_coords.extend([int(layer), int(x), int(y)])

        self.reward_vector = reward.reshape(-1)

    def mutate(self):
        return self.reward_parameters + np.random.randn(*self.reward_parameters.shape) / 10


def maybe_multiprocess(population):
    if config.DEBUG:
        with multiprocessing.Pool() as pool:
            retr = pool.map(fitness_function, population)
    else:
        retr = [fitness_function(p) for p in population]

    population_fitness, options = zip(*retr)
    return population_fitness


class Search:
    def __init__(self, population_size=config.population):
        print("init")
        self.population_size = population_size
        _e = config.environment()
        obs = _e.reset()
        reward_space_size = obs.reshape(-1).shape[0]
        self.reward_space_size = 3  # reward_space_size
        print("init done")

    def optimize(self):
        # self.test_intuitive_cases(self.reward_space_size)
        # if config.DEBUG:
        #     _ = fitness_function(ConstrainedReward(np.random.randn(self.reward_space_size)))  # raise errors

        population = self.generate_initial_population()

        for optimization_iteration in tqdm.tqdm(range(config.evolution_iters)):
            population_fitness = self.calculate_fitness(population)

            best_sample, best_fitness = max(zip(population, population_fitness), key=lambda x: x[1])
            self.mutate(best_sample, population)

            config.tensorboard.add_scalar('optimization/mean', np.array(population_fitness).mean(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/best', best_fitness, optimization_iteration)

        return None, None

    def calculate_fitness(self, population):
        population_fitness = maybe_multiprocess(population)

        # population_fitness, options = zip(*[fitness_function(p) for p in population])
        print("")
        print("==RESULTS==")
        for r, fit in zip(population, population_fitness):
            print(f"{r.reward_coords} achieved fit {fit}")
        return population_fitness

    def mutate(self, best_sample, population):
        for idx, p in enumerate(population):
            if p == best_sample:
                continue
            new_parameters = best_sample.mutate()
            population[idx] = ConstrainedReward(new_parameters)
            print('newly generated reward params', population[idx].reward_parameters)

    def generate_initial_population(self):
        population = [ConstrainedReward(np.random.randn(self.reward_space_size)) for _ in range(config.population)]
        population[0] = ConstrainedReward(np.array([
            C.DOOR_LAYER, (config.agent_view_size - 2), (config.agent_view_size // 2)
        ]), normalize=False)
        print("==INITIAL POPULATION==")
        for r in population:
            print(f"{r.reward_coords}")
        return population

    @staticmethod
    def test_intuitive_cases(reward_space_size):
        # Search.test_door(reward_space_size)
        Search.test_open_door(config.Minigrid.nr_layers * config.agent_view_size * config.agent_view_size)
        # Search.test_door_gradient(reward_space_size)

    @staticmethod
    def test_door_and_goal(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ . _ _  1 _ _ . _ _
        # 2 _ _ . 1 <  2 _ _ . 1 <
        # 3 _ _ . _ _  3 _ _ . _ _
        # 4 _ _ _ _ _  4 _ _ _ _ _
        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.005
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, config.agent_view_size // 2, -2] = 1
        intrinsic_motivation[C.DOOR_LAYER, :, -3] = 0.1

        intrinsic_motivation[C.FOOD_LAYER, config.agent_view_size // 2, -2] = 2
        intrinsic_motivation[C.FOOD_LAYER, :, -3] = 0.2

        door_fitness, _ = fitness_function(intrinsic_motivation.reshape(-1))  # raise errors
        print('fitness with door and goal curiosity', door_fitness, intrinsic_motivation.reshape(-1), sep='\n')

    @staticmethod
    def test_open_door(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ _ _ _  1 _ _ _ _ _
        # 2 _ _ _ _ _  2 _ _ _ _ _
        # 3 _ _ _ _ _  3 _ _ _ _ _
        # 4 _ 1 * 1 _  4 _ _ _ _ _

        agent_col = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.00001
        # intrinsic_motivation = np.ones(shape=reward_space_size) * -0.00001
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, -2, agent_col] = 1.1
        # intrinsic_motivation[C.UNWALKABLE_LAYER, -1, agent_col + 1] = 0.6
        # intrinsic_motivation[C.UNWALKABLE_LAYER, -1, agent_col - 1] = 0.6

        reward = Reward(reward_vector=intrinsic_motivation.reshape(-1))
        door_fitness, _ = fitness_function(reward)  # raise errors
        print('fitness with open doors curiosity', door_fitness, intrinsic_motivation, sep='\n')

    @staticmethod
    def test_door_gradient(reward_space_size):
        pass

    @staticmethod
    def test_door(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ _ _ _  1 _ _ _ _ _
        # 2 _ _ _ _ _  2 _ _ _ _ _
        # 3 _ _ _ _ _  3 _ _ _ _ _
        # 4 _ 1 * 1 _  4 _ _ _ _ _

        agent_col = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.0000001
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, -2, agent_col] = 0.6

        reward = Reward(reward_vector=intrinsic_motivation.reshape(-1))
        door_fitness, _ = fitness_function(reward)  # raise errors
        print('fitness with open doors curiosity', door_fitness, intrinsic_motivation, sep='\n')

    def get_best_option_score(self):
        # options = [learners.approx_q_learning.generate_option(config.environment(), (4, 4, 0), False), ]
        # options = [learners.approx_q_learning.generate_option(config.environment(), (6, 6, 0), False), ]
        print("eval options", str(sorted([str(o) for o in options]))[:50], end=' ')
        fitness = shared.eval_options(envs.minigrid.MiniGrid(), options)
        print(config.option_eval_training_steps, fitness)
        return fitness

    @staticmethod
    def eval_solutions(_fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(_fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options


def main():
    regressor = Search()
    regressor.optimize()


if __name__ == "__main__":
    main()
