import math
from typing import Callable, Tuple, List, Dict, Iterator, Collection

from schema import Schema, And, Optional, Or

from TP2.character import Character
from TP2.config import Config, Param, ParamValidator
import random

CoupleSelector = Callable[[Collection[Character]], Collection[Tuple[Character, Character]]]
InternalCoupleSelector = Callable[[Collection[Character], int, Param], Collection[Tuple[Character, Character]]]


def _extract_coupling_selector_params(config: Config) -> Param:
    return Config.validate_param(config.parent_coupling, Schema({
        Optional('couple_count', default=-1): And(int, lambda count: count > 0),  # Default is parent_count//2
        'method': {
            'name': And(str, Or(*tuple(_couple_selector_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_couple_selector(config: Config) -> CoupleSelector:
    coupling_selector_params: Param = _extract_coupling_selector_params(config)

    method, coupling_params_schema = _couple_selector_dict[coupling_selector_params['method']['name']]
    couple_count: int = coupling_selector_params['couple_count']
    method_params: Param = coupling_selector_params['method']['params']
    if coupling_params_schema:
        method_params = Config.validate_param(method_params, coupling_params_schema)

    return lambda parents: method(parents, couple_count if couple_count > 0 else len(parents)//2, method_params)


# TODO(tobi): Mejorable
def equitable_random_coupling(parents: Collection[Character], couple_count: int, coupling_params: Param) -> Collection[Tuple[Character, Character]]:
    parents = list(parents)  # Preserve original list
    natural_coupling_count: int = len(parents)//2
    ret: List[Tuple[Character, Character]] = []

    if len(parents) % 2 != 0:
        parents.pop(random.randint(0, len(parents) - 1))

    while couple_count >= natural_coupling_count:
        random.shuffle(parents)
        it: Iterator[Character] = iter(parents)
        ret.extend(zip(it, it))
        couple_count -= natural_coupling_count

    if couple_count == 0:
        return ret

    random.shuffle(parents)
    it = iter(parents)
    for couple in zip(it, it):
        ret.append(couple)
        couple_count -= 1
        if couple_count == 0:
            break

    return ret


def _random_couple_from_population(population: Collection[Character]) -> Tuple[Character, Character]:
    # TODO(tobi): Borrar casteo a lista. Probablemente cambiar todo a List
    couple = random.sample(list(population), 2)
    return couple[0], couple[1]


def chaotic_random_coupling(parents: Collection[Character], couple_count: int, coupling_params: Param) -> Collection[Tuple[Character, Character]]:
    return [_random_couple_from_population(parents) for _ in range(math.floor(couple_count / 2))]


_couple_selector_dict: Dict[str, Tuple[InternalCoupleSelector, ParamValidator]] = {
    'equitable_random': (equitable_random_coupling, None),
    'chaotic_random': (chaotic_random_coupling, None),
}
