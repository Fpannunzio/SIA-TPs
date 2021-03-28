import sys
from typing import Dict, Any, List, Optional

import yaml
from schema import Schema, SchemaError, And, Or

import character

Param = Dict[str, Any]
ParamValidator = Optional[Schema]


def print_error_and_exit(error_msg: str, exit_code: int):
    print(error_msg)
    sys.exit(exit_code)


class Config:

    @staticmethod
    def validate_param(param: Param, schema: Schema) -> Param:
        try:
            return schema.validate(param)
        except SchemaError as e:
            sys.exit(e.code)

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

        valid_class_types: List[str] = [e.value for e in character.CharacterType]

        args = Config.validate_param(args, Schema({
            'population_size': And(int, lambda population_size: 0 < population_size < 990000),
            'class': And(str, Or(*tuple(e.value for e in character.CharacterType))),
            'item_files': dict,
            'parent_selection': dict,
            'parent_coupling': dict,
            'crossover': dict,
            'mutation': dict,
            'survivor_selection': dict,
            'recombination': dict,
            'end_condition': dict,
        }, ignore_extra_keys=True))

        self.character_class: str = args['class']
        self.population_size: int = args['population_size']
        self.item_files: Param = args['item_files']
        self.parent_selection: Param = args['parent_selection']
        self.parent_coupling: Param = args['parent_coupling']
        self.crossover: Param = args['crossover']
        self.mutation: Param = args['mutation']
        self.survivor_selection: Param = args['survivor_selection']
        self.recombination: Param = args['recombination']
        self.end_condition: Param = args['end_condition']
