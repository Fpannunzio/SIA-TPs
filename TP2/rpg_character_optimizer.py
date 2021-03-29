import sys

from config import Config
from engine import Engine
from generation import Generation

from items import ItemRepositories


def main(config_file: str):

    # Load Config from config_file
    config: Config = Config(config_file)

    # Load Items from .tsv Files
    item_repositories: ItemRepositories = ItemRepositories(config.item_files)

    # Configure Simulation
    engine: Engine = Engine(config, item_repositories)

    # Start Simulation
    last_generation: Generation = engine.resolve_simulation()

    print(f'Total Simulation Iterations: {last_generation.gen_count}\n'
          f'Best Character from Simulation: {last_generation.get_best_character()}\n')

    print('Done')

    sys.exit(0)


# Usage: python3 rpg_character_optimizer.py [config_file_path]
if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    main(config_file)
