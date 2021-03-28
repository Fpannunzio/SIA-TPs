from typing import Callable, Collection, List, Dict, Tuple

from schema import Schema, And, Optional, Or

from TP2.character import Character
from TP2.config import Config, Param, ParamValidator
from TP2.generation import Generation
from TP2.selection import SurvivorSelector

Recombiner = Callable[[Generation, List[Character], SurvivorSelector], Collection[Character]]
InternalRecombiner = Callable[[Generation, List[Character], SurvivorSelector, Param], Collection[Character]]


def _extract_recombiner_params(config: Config) -> Param:
    return Config.validate_param(config.recombination, Schema({
        'method': {
            'name': And(str, Or(*tuple(_recombiner_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_recombiner(config: Config) -> Recombiner:
    recombiner_params: Param = _extract_recombiner_params(config)
    method, recombiner_method_params_schema = _recombiner_dict[recombiner_params['method']['name']]
    recombiner_method_params: Param = recombiner_params['method']['params']
    if recombiner_method_params_schema:
        recombiner_method_params = Config.validate_param(recombiner_method_params, recombiner_method_params_schema)

    return lambda current_generation, children, survivor_selector: \
        method(current_generation, children, survivor_selector, recombiner_method_params)


def fill_all_selection(current_generation: Generation, children: List[Character],
                       survivor_selection: SurvivorSelector, recombiner_params: Param) -> Collection[Character]:
    return survivor_selection(
        Generation(current_generation.population + children, current_generation.gen_count),
        len(current_generation.population)
    )


def fill_parent_selection(current_generation: Generation, children: List[Character],
                          survivor_selection: SurvivorSelector, recombiner_params: Param) -> Collection[Character]:
    k: int = len(children)
    n: int = len(current_generation.population)

    if k > n:
        return survivor_selection(Generation(children, current_generation.gen_count), n)
    else:
        return survivor_selection(current_generation, n-k) + children


_recombiner_dict: Dict[str, Tuple[InternalRecombiner, ParamValidator]] = {
    'fill_all': (fill_all_selection, None),
    'fill_parent': (fill_parent_selection, None),
}