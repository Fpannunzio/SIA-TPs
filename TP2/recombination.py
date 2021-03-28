from typing import Callable, Dict, Tuple

from schema import Schema, And, Optional, Or

from TP2.config import Config, Param, ParamValidator
from TP2.crossover import Children
from TP2.generation import Generation
from TP2.selection import SurvivorSelector

# Exported Types
Recombiner = Callable[[Generation, Children, SurvivorSelector], Generation]

# Internal Types
InternalRecombiner = Callable[[Generation, Children, SurvivorSelector, Param], Generation]


def _extract_recombiner_params(recombiner_params: Param) -> Param:
    return Config.validate_param(recombiner_params, Schema({
        'method': {
            'name': And(str, Or(*tuple(_recombiner_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_recombiner(recombiner_params: Param) -> Recombiner:
    recombiner_params = _extract_recombiner_params(recombiner_params)

    method, recombiner_method_params_schema = _recombiner_dict[recombiner_params['method']['name']]
    recombiner_method_params: Param = recombiner_params['method']['params']
    if recombiner_method_params_schema:
        recombiner_method_params = Config.validate_param(recombiner_method_params, recombiner_method_params_schema)

    return lambda current_generation, children, survivor_selector: \
        method(current_generation, children, survivor_selector, recombiner_method_params)


def fill_all_selection(current_generation: Generation, children: Children,
                       survivor_selector: SurvivorSelector, recombiner_params: Param) -> Generation:

    temp_gen: Generation = Generation(current_generation.population + children, current_generation.gen_count)
    return current_generation.create_next_generation(survivor_selector(temp_gen, len(current_generation)))


def fill_parent_selection(current_generation: Generation, children: Children,
                          survivor_selector: SurvivorSelector, recombiner_params: Param) -> Generation:
    k: int = len(children)
    n: int = len(current_generation)

    if k > n:
        return current_generation.create_next_generation(survivor_selector(Generation(children, current_generation.gen_count), n))
    else:
        return current_generation.create_next_generation(survivor_selector(current_generation, n - k) + children)


_recombiner_dict: Dict[str, Tuple[InternalRecombiner, ParamValidator]] = {
    'fill_all': (fill_all_selection, None),
    'fill_parent': (fill_parent_selection, None),
}