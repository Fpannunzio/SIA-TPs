import math
from typing import Callable, Tuple, List, Dict, Iterator

from schema import Schema, And, Optional, Or

from TP2.character import Character
from TP2.config import Config, Param, ParamValidator
import random

from TP2.selection import Parents

# Exported Types
Couple = Tuple[Character, Character]
Couples = List[Couple]
CouplesSelector = Callable[[Parents], Couples]

# Internal Types
InternalCoupleSelector = Callable[[Parents, int, Param], Couples]


def _validate_coupling_selector_params(coupling_selector_params: Param) -> Param:
    return Config.validate_param(coupling_selector_params, Schema({
        Optional('couple_count', default=-1): And(int, lambda count: count > 0),  # Default is parent_count//2
        'method': {
            'name': And(str, Or(*tuple(_couple_selector_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_couples_selector(coupling_selector_params: Param) -> CouplesSelector:
    coupling_selector_params = _validate_coupling_selector_params(coupling_selector_params)

    method, coupling_params_schema = _couple_selector_dict[coupling_selector_params['method']['name']]
    couple_count: int = coupling_selector_params['couple_count']
    method_params: Param = coupling_selector_params['method']['params']
    if coupling_params_schema:
        method_params = Config.validate_param(method_params, coupling_params_schema)

    return lambda parents: method(parents, couple_count if couple_count > 0 else len(parents)//2, method_params)


# TODO(tobi): Mejorable
def equitable_random_coupling(parents: Parents, couple_count: int, coupling_params: Param) -> Couples:
    parents = parents.copy()  # Preserve original list
    natural_coupling_count: int = len(parents)//2
    ret: Couples = []

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


def _get_random_couple(parents: Parents) -> Couple:
    couple = random.sample(parents, 2)
    return couple[0], couple[1]


def chaotic_random_coupling(parents: Parents, couple_count: int, coupling_params: Param) -> Couples:
    return [_get_random_couple(parents) for _ in range(math.floor(couple_count / 2))]


_couple_selector_dict: Dict[str, Tuple[InternalCoupleSelector, ParamValidator]] = {
    'equitable_random': (equitable_random_coupling, None),
    'chaotic_random': (chaotic_random_coupling, None),
}
