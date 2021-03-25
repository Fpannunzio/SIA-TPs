from typing import Dict, Any, Optional

import yaml

StrategyParams = Optional[Dict[str, Any]]


class Config:

    def __init__(self, config_path: str):

        try:
            stream = open(config_path, 'r')  # 'config.yaml' contains a single YAML document.
        except FileNotFoundError:
            raise ValueError(f'Config file missing. Make sure "{config_path}" is present')

        try:
            args = yaml.safe_load(stream)
        except Exception:
            raise ValueError(f'There was a problem parsing the configuration file {config_path}. Make sure syntax is '
                             f'appropriate')

        if 'items_files' not in args:
            raise ValueError(f'There are arguments missing. Make sure "items_files" is present')

        self.item_files: Dict[str, Any] = args['item_files']

    # def __repr__(self) -> str:
    #     return f'Level: {repr(self.level)}. ' \
    #            f'Strategy: {repr(self.strategy)}. ' \
    #            f'Strategy Params: {repr(self.strategy_params)}'
