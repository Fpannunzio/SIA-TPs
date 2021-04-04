import sys
import random
import time
from typing import Optional

from plot import AsyncPlotter, get_plotter
from config import Config
from engine import Engine
from generation import Generation

from items import ItemRepositories


def append_result_to_output_file(output_path: Optional[str], result: str) -> None:
    if output_path is None:
        return

    print(f'Writing output to {output_path}')
    try:
        out = open(output_path, 'a')
        out.write(result)
        out.close()
    except FileNotFoundError:
        print(f'Warning: Could not open or create file {output_path}. Output is not being recorded')


def main(config_file: str):

    print('----------------- Welcome to RPG Character Optimizer -------------------')

    # Load Config from config_file
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    print(f'You have selected to optimize the {config.character_class} class')

    # Initialize Application Seed
    seed = config.seed if config.seed else int(time.time())
    random.seed(seed)

    # Load Items from .tsv Files
    print('Loading Items...')
    item_repositories: ItemRepositories = ItemRepositories(config.item_files)

    # Load Plotters
    plotter: AsyncPlotter = get_plotter(config.plotting)

    # Configure Simulation
    print('Verifying correct configuration...')
    engine: Engine = Engine(config, item_repositories, plotter)

    print(f'Starting Simulation (seed: {seed})')

    try:
        # Start Simulation
        last_generation: Generation = engine.resolve_simulation()

        print(f'Simulation Ended')

        # TODO(tobi) imprimir best character historico no de la ultima generacion
        simulation_output: str = f'---------------------- Simulation Output --------------------------\n' \
                                 f'Simulation Seed: {seed}\n' \
                                 f'Total Simulation Iterations: {last_generation.gen_count}\n' \
                                 f'Best {config.character_class} from Simulation: {last_generation.get_best_character()}\n'

        # Output Final Info
        append_result_to_output_file(config.output_file, simulation_output)
        print()
        print(simulation_output)
        print()

        # Wait for plotter to end
        print('Plotting data...')
        plotter.wait()

    except (KeyboardInterrupt, Exception) as e:
        plotter.kill()
        raise e

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
        sys.exit(0)

    except (ValueError, FileNotFoundError) as ex:
        print('\nAn Error Was Found!!')
        print(ex)
        sys.exit(1)

    except Exception as ex:
        print('An unexpected error occurred')
        raise ex
