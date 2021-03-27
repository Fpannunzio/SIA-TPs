from typing import Callable, Collection, List, Dict


from TP2.character import Character, Generation
from TP2.config_loader import Config
from TP2.selection import SurvivorSelector

Recombination = Callable[[List[Character], List[Character], int, SurvivorSelector], Collection[Character]]


def fill_all_selection(current_generation: List[Character], children: List[Character], generation_number: int, survivor_selection: SurvivorSelector) -> Collection[Character]:
    return survivor_selection(Generation(current_generation + children, generation_number), len(current_generation))


def fill_parent_selection(current_generation: List[Character], children: List[Character], generation_number: int, survivor_selection: SurvivorSelector) -> Collection[Character]:
    k = len(children)
    n = len(current_generation)

    if k > n:
        return survivor_selection(Generation(children, generation_number), n)
    else:
        return survivor_selection(Generation(current_generation, generation_number), n-k) + children


def get_recombination_impl(config: Config) -> Recombination:
    #TODO sacar del config el metodo de implementacion
    return survivor_selection_impl_dict['fill_all_selection']


survivor_selection_impl_dict: Dict[str, Recombination] = {
    'fill_all_selection': fill_all_selection,
    'fill_parent_selection': fill_parent_selection,
}