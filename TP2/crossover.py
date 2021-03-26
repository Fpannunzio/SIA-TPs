from typing import Callable, Collection, Tuple, List, Dict, Union, Iterator, NamedTuple

from TP2.character import Character, CharacterType
from TP2.config_loader import Config
import random

from TP2.items import Item, ItemSet

Crossover = Callable[[Collection[Tuple[Character, Character]]], Collection[Tuple[Character, Character]]]

Gene = NamedTuple('Gene', [('accessor', Callable[[Character], Union[float, Item]]), ('name', str)])

gen_accessors: List[Callable[[Character], Union[float, Item]]] = [
    lambda character: character.height,
    lambda character: character.items.weapon,
    lambda character: character.items.boots,
    lambda character: character.items.helmets,
    lambda character: character.items.gauntlets,
    lambda character: character.items.chestpieces
]

gen_names: List[str] = [
    'height', 'weapon', 'boots', 'helmets', 'gauntlets', 'chestpieces'
]


def get_gen(character: Character, index: int) -> Union[float, Item]:
    return gen_accessors[index](character)


def get_crossover_impl(config: Config) -> Crossover:
    # TODO por ahora solo esta random coupling
    return crossover_impl_dict['single_point']


def single_point(parents: Collection[Tuple[Character, Character]]) -> Collection[Tuple[Character, Character]]:
    return list(map(single_point_swap, parents))


def single_point_swap(couple: Tuple[Character, Character]) -> Tuple[Character, Character]:
    p: int = random.randrange(len(gen_accessors) - 2)  # la ultima opcion es equivalente a duplicar los padres

    character_type: CharacterType = couple[0].type

    child1: List[Union[float, Item]] = []
    child2: List[Union[float, Item]] = []

    for i in range(len(gen_accessors)):
        parent: int = 0

        if i > p:
            parent = 1

        child1.append(get_gen(couple[parent], i))
        child2.append(get_gen(couple[1 - parent], i))

    itemSet1: ItemSet = ItemSet(child1[1], child1[2], child1[3], child1[4], child1[5])
    itemSet2: ItemSet = ItemSet(child2[1], child2[2], child2[3], child2[4], child2[5])

    return Character(character_type, child1[0], itemSet1), Character(character_type, child2[0], itemSet2)

def child_creation(couple: Tuple[Character, Character], parent_sequence: Iterator[int]) -> Tuple[Character, Character]:

    children_dicts: Tuple[Dict[str, Union[float, Item]]] = ({}, {})

    for parent in parent_sequence:
        children_dicts[0][]

crossover_impl_dict: Dict[str, Crossover] = {
    'single_point': single_point,
}


def get_single_point_parent_seq(gene_count: int) -> Iterator[int]:
    p: int = random.randrange(gene_count - 1)
    return map(lambda i: 0 if i <= p else 1, range(gene_count))
