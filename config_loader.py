from typing import Dict, Any, Optional

import yaml


class Config:

    def __init__(self, config_path: str):

        try:
            stream = open(config_path, 'r')  # 'config.yaml' contains a single YAML document.
        except FileNotFoundError:
            raise RuntimeError(f'Config file missing. Make sure "{config_path}" is present')

        try:
            args = yaml.safe_load(stream)
        except Exception:
            raise RuntimeError(f'There was a problem parsing the configuration file {config_path}. Make sure syntax is '
                               f'appropriate')

        if 'level' not in args or 'strategy' not in args or 'name' not in args['strategy']:
            raise RuntimeError(f'There are arguments missing. Make sure "level" and "strategy: name" are present')

        self.level: str = args['level']
        self.strategy: str = args['strategy']['name']
        self.strategy_params: Optional[Dict[str, Any]] = args['strategy'].get('params', None)
        self.render: bool = args.get('render', True)

    def __repr__(self) -> str:
        return f'Level: {self.level}. ' \
               f'Strategy: {self.strategy}. ' \
               f'Strategy Params: {self.strategy_params}'
