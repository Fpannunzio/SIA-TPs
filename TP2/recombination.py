from typing import Callable, Collection, List, Dict


from TP2.character import Character
from TP2.config_loader import Config
from TP2.generation import Generation
from TP2.selection import SurvivorSelector

Recombination = Callable[[Generation, List[Character], SurvivorSelector], Collection[Character]]


def fill_all_selection(current_generation: Generation, children: List[Character], survivor_selection: SurvivorSelector) -> Collection[Character]:
    return survivor_selection(Generation(current_generation.characters + children, current_generation.generation), len(current_generation.characters))


def fill_parent_selection(current_generation: Generation, children: List[Character], survivor_selection: SurvivorSelector) -> Collection[Character]:
    k = len(children)
    n = len(current_generation.characters)

    if k > n:
        return survivor_selection(Generation(children, current_generation.generation), n)
    else:
        return survivor_selection(current_generation, n-k) + children


def get_recombination_impl(config: Config) -> Recombination:
    #TODO sacar del config el metodo de implementacion
    return survivor_selection_impl_dict['fill_all_selection']


survivor_selection_impl_dict: Dict[str, Recombination] = {
    'fill_all_selection': fill_all_selection,
    'fill_parent_selection': fill_parent_selection,
}