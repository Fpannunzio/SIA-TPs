import itertools
import random
from typing import Callable, Collection, Tuple, List, Dict, Union, Iterator, NamedTuple

import numpy as np

from TP2.character import Character, CharacterType
from TP2.config_loader import Config
from TP2.items import Item, ItemSet

Crossover = Callable[[Collection[Tuple[Character, Character]]], Collection[Character]]
ParentSeqGenerator = Callable[[int], Iterator[int]]


gene_getters: Dict[str, Callable[[Character], Union[float, Item]]] = {
    'height': lambda character: character.height,
    'weapon': lambda character: character.items.weapon,
    'boots': lambda character: character.items.boots,
    'helmets': lambda character: character.items.helmets,
    'gauntlets': lambda character: character.items.gauntlets,
    'chestpieces': lambda character: character.items.chestpieces
}

gene_names: Collection[str] = sorted(gene_getters.keys())


def get_gen(character: Character, gene: str) -> Union[float, Item]:
    return gene_getters[gene](character)


def get_crossover_impl(config: Config) -> Crossover:
    # TODO valor hardcodeado
    def parent_seq_gen(): return parent_seq_dict['single_point'](len(gene_names))

    flatten = itertools.chain.from_iterable

    return lambda parents: list(flatten(map(lambda couple: child_creation(couple, parent_seq_gen), parents)))


def child_creation(couple: Tuple[Character, Character], parent_seq_gen: Callable[[], Iterator[int]]) -> List[Character]:
    children_genes: Tuple[Dict[str, Union[float, Item]], Dict[str, Union[float, Item]]] = {}, {}

    parent_sequence: Iterator[int] = parent_seq_gen()

    for gene in gene_names:
        parent = next(parent_sequence)
        children_genes[0][gene] = get_gen(couple[parent], gene)
        children_genes[1][gene] = get_gen(couple[1 - parent], gene)

    children_item_sets: Tuple[ItemSet, ItemSet] = \
        item_set_from_gene_map(children_genes[0]), item_set_from_gene_map(children_genes[1])

    children_type: CharacterType = couple[0].type

    return [Character(children_type, children_genes[0]['height'], children_item_sets[0]),
            Character(children_type, children_genes[1]['height'], children_item_sets[1])]


def item_set_from_gene_map(gene_map: Dict[str, Union[float, Item]]) -> ItemSet:
    return ItemSet(gene_map['weapon'], gene_map['boots'], gene_map['helmets'],
                   gene_map['gauntlets'], gene_map['chestpieces'])


def get_k_point_parent_seq(gene_count: int, k: int) -> Iterator[int]:
    p = np.array(random.sample(range(gene_count + 1), k)).sort()  # [0, gene_count + 1)
    return map(lambda i: np.searchsorted(p, i, side='right') % 2, range(gene_count))


def get_single_point_parent_seq(gene_count: int) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 1)


def get_two_point_parent_seq(gene_count: int) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 2)


def uniform_parent_seq(gene_count: int) -> Iterator[int]:
    return map(lambda i: random.randint(0, 1), range(gene_count))


parent_seq_dict: Dict[str, ParentSeqGenerator] = {
    'single_point': get_single_point_parent_seq,
    'two_point': get_two_point_parent_seq,
    # 'annular': get_two_point_parent_seq, EXISTE???
    'uniform': uniform_parent_seq,
}
