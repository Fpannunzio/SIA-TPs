import sys

from TP2.config_loader import Config
from TP2.item_loader import Items


def main(config_file: str):

    # Load Config from config.yaml
    config: Config = Config(config_file)

    # Load Items from .tsv Files
    items: Items = Items(config)


# Usage: python3 rpg_character_optimizer.py [config_file_path]
if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    # try:
    main(config_file)

    # except ValueError as e:
    #     print('-' * 50)
    #     print(f'There was an error found in the configuration file {config_file} or in the level file selected:')
    #     print(e)
    #
    # except FileNotFoundError as e:
    #     print('-' * 50)
    #     print(f'Config file or level file {e.filename} was not found')
    #
    # except RuntimeError as e:
    #     print('-' * 50)
    #     print('An unexpected error was encountered. Please inform the developers about this issue.')
    #     raise e
