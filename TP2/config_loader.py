from typing import Dict, Any

import yaml


class Config:

    def __init__(self, config_path: str):

        try:
            stream = open(config_path, 'r')  # contains a single YAML document.
        except FileNotFoundError:
            raise ValueError(f'Config file missing. Make sure "{config_path}" is present')

        try:
            args = yaml.safe_load(stream)
        except Exception:
            raise ValueError(f'There was a problem parsing the configuration file {config_path}. Make sure syntax is '
                             f'appropriate')

        if 'item_files' not in args:
            raise ValueError(f'There are arguments missing. Make sure "item_files" is present')

        if 'gen_size' not in args:
            raise ValueError(f'There are arguments missing. Make sure "gen_size" is present')

        if 'k' not in args:
            raise ValueError(f'There are arguments missing. Make sure "k" is present')

        self.item_files: Dict[str, Any] = args['item_files']
        self.character_class: str = args['class']
        self.gen_size: int = args['gen_size']
        self.k: int = args['k']

    # def __repr__(self) -> str:
    #     return f'Level: {repr(self.level)}. ' \
    #            f'Strategy: {repr(self.strategy)}. ' \
    #            f'Strategy Params: {repr(self.strategy_params)}'
