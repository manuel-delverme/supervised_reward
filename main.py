import config
import controller.genetic
import controller.hyperband
import controller.meta_controller


def main():
    regressor = controller.genetic.GeneticEvolution()
    regressor.optimize(config.experiment_id)


if __name__ == "__main__":
    main()
