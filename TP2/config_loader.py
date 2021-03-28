import sys
from typing import Dict, Any

import yaml
from schema import Schema, SchemaError

Param = Dict[str, Any]


def print_and_exit_with_error(message: str, error_code: int):
    print(message)
    sys.exit(error_code)


class Config:

    @staticmethod
    def validate_param(param: Param, schema: Schema) -> Param:
        try:
            return schema.validate(param)
        except SchemaError as e:
            print_and_exit_with_error(str(e), 2)

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

        args = Config.validate_param(args, Schema({
            'gen_size': int,
            'class': str,
            'item_files': dict,
            'parent_selection': dict,
            'end_condition': dict,
            'k': int,
        }))

        self.character_class: str = args['class']
        self.gen_size: int = args['gen_size']
        self.item_files: Param = args['item_files']
        self.parent_selection: Param = args['parent_selection']
        self.end_condition: Param = args['end_condition']
        self.k: int = args['k']
