import sys

from plot import AsyncPlotter, get_plotter
from config import Config
from engine import Engine
from generation import Generation

from items import ItemRepositories


def main(config_file: str):

    # Load Config from config_file
    config: Config = Config(config_file)

    # Load Items from .tsv Files
    item_repositories: ItemRepositories = ItemRepositories(config.item_files)

    # Load Plotters
    plotter: AsyncPlotter = get_plotter(config.plotting)

    # Configure Simulation
    engine: Engine = Engine(config, item_repositories, plotter)

    print('Starting Simulation...')

    try:
        # Start Simulation
        last_generation: Generation = engine.resolve_simulation()

        # Print Final Info
        print(f'Total Simulation Iterations: {last_generation.gen_count}\n'
              f'Best Character from Simulation: {last_generation.get_best_character()}\n')

        # Wait for plotter to end
        plotter.wait()

    except KeyboardInterrupt:
        plotter.kill()
        sys.exit(1)

    print('Done')

    sys.exit(0)


# Usage: python3 rpg_character_optimizer.py [config_file_path]
if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    try:
        main(config_file)
    except KeyboardInterrupt:
        print('sigint')
