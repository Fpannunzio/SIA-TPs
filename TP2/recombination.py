from typing import Callable, Collection, List, Dict

import numpy as np

from TP2.character import Character
from TP2.config_loader import Config
from TP2.selection import Selection

Recombination = Callable[[List[Character], List[Character], Selection], List[Character]]


def fill_all_selection(parents: List[Character], children: List[Character], selection: Selection) -> List[Character]:
    return selection(parents + children, len(parents))


def fill_parent_selection(parents: List[Character], children: List[Character], selection: Selection) -> List[Character]:
    k = len(children)
    n = len(parents)

    if k > n:
        return selection(children, n)
    else:
        return selection(parents, n-k) + children


def get_recombination_impl(config: Config) -> Recombination:
    return survivor_selection_impl_dict['fill_all_selection']


survivor_selection_impl_dict: Dict[str, Recombination] = {
    'fill_all_selection': fill_all_selection,
    'fill_parent_selection': fill_parent_selection,
}