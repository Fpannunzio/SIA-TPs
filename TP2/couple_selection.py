import math
from typing import Callable, Collection, Tuple, List, Dict

from TP2.character import Character
from TP2.config_loader import Config
import random

ParentSelection = Callable[[Collection[Character], int], Collection[Tuple[Character, Character]]]


def get_couple_selection_impl(config: Config) -> ParentSelection:
    # TODO por ahora solo esta random coupling
    return parent_selection_impl_dict['random_coupling']


def random_coupling(parents: Collection[Character], reproduction_factor: int) -> \
        Collection[Tuple[Character, Character]]:
    parents = list(parents)

    return [random_couple_from_population(parents) for _ in range(math.floor(reproduction_factor / 2))]


def random_couple_from_population(population: List[Character]) -> Tuple[Character, Character]:
    couple = random.sample(population, 2)

    return couple[0], couple[1]


parent_selection_impl_dict: Dict[str, ParentSelection] = {
    'random_coupling': random_coupling,
}
