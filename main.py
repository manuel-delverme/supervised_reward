import config
import controller.genetic
import controller.hyperband
import controller.meta_controller


def main():
    reward_space_size = (config.agent_view_size * config.agent_view_size) + 1
    regressor = controller.genetic.GeneticEvolution(reward_space_size=reward_space_size, )
    regressor.optimize(config.experiment_id)


if __name__ == "__main__":
    main()
