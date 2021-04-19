import sys
from typing import Dict, Any

import typing
import yaml
from schema import Schema, SchemaError, Optional, And

Param = Dict[str, Any]
ParamValidator = typing.Optional[Schema]


def print_error_and_exit(error_msg: str, exit_code: int):
    print(error_msg)
    sys.exit(exit_code)


class Config:

    @staticmethod
    def validate_param(param: Param, schema: Schema) -> Param:
        try:
            return schema.validate(param)
        except SchemaError as e:
            print('A problem was found on the configuration file:\n')
            sys.exit(e.code)

    def __init__(self, config_path: str):

        try:
            stream = open(config_path, 'r')  # contains a single YAML document.
        except FileNotFoundError:
            raise FileNotFoundError(f'Config file missing. Make sure "{config_path}" is present')

        try:
            args = yaml.safe_load(stream)
        except Exception:
            raise ValueError(f'There was a problem parsing the configuration file {config_path}. Make sure syntax is '
                             f'appropriate')

        args = Config.validate_param(args, Schema({
            'training_set': {
                'inputs': str,
                Optional('input_line_count', default=1): And(int, lambda i: i > 0),
                'outputs': str,
                Optional('output_line_count', default=1): And(int, lambda i: i > 0),
                Optional('normalize_values', default=False): bool,
            },
            'validation_set': {
                'inputs': str,
                'outputs': str,
                Optional('normalize_values', default=False): bool,
            },
            'network': dict,
        }, ignore_extra_keys=True))

        self.training_set: Param = args['training_set']
        self.validation_set: Param = args['validation_set']
        self.network: Param = args['network']
