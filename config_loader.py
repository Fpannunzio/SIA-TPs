import yaml


class Config(object):

    def __init__(self):

        try:
            stream = open('config.yaml', 'r')  # 'document.yaml' contains a single YAML document.
        except FileNotFoundError:
            raise RuntimeError(f'Config file missing. Make sure "config.yaml" is present')

        try:
            args = yaml.safe_load(stream)
        except Exception:
            raise RuntimeError(f'There was a problem parsing the configuration file. Make sure syntax is appropriate')

        if 'level' not in args or 'strategy' not in args or 'name' not in args['strategy']:
            raise RuntimeError(f'There are arguments missing. Make sure "level" and "strategy: name" are present')

        self.level = args['level']
        self.strategy = args['strategy']['name']
        self.strategy_params = args['strategy']['params'] if 'params' in args['strategy'] else None

    def __repr__(self) -> str:
        return f'Level: {self.level}. ' \
               f'Strategy: {self.strategy}. ' \
               f'Strategy Params: {self.strategy_params}'
