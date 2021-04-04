import itertools
import random
from typing import Callable, Tuple, Dict, Union, Iterator

import numpy as np
from schema import Schema, And, Optional, Or

from character import Character, CharacterType
from config import Config, Param, ParamValidator
from couple_selection import Couples, Couple
from generation import Population
from items import Item, ItemSet

# Exported Types
Children = Population
Crossover = Callable[[Couples], Children]

# Internal Types
ParentSeqGenerator = Callable[[int, Param], Iterator[int]]


def _validate_crossover_params(crossover_params: Param) -> Param:
    return Config.validate_param(crossover_params, Schema({
        Optional('children_eq_parents_prob', default=0): And(float, lambda p: 0 <= p <= 1),
        Optional('children_per_couple', default=2): And(int, lambda count: count > 0),
        'method': {
            'name': And(str, Or(*tuple(_parent_seq_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_crossover(crossover_params) -> Crossover:
    crossover_params = _validate_crossover_params(crossover_params)

    method, crossover_method_param_schema = _parent_seq_dict[crossover_params['method']['name']]
    crossover_method_params: Param = crossover_params['method']['params']
    if crossover_method_param_schema:
        crossover_method_params = Config.validate_param(crossover_method_params, crossover_method_param_schema)
    # TODO(tobi): Ver que onda children_per_couple. Como implementarlo
    children_per_couple: int = crossover_params['children_per_couple']
    pc: int = crossover_params['children_eq_parents_prob']

    def parent_seq_gen(): return method(len(Character.gene_list), crossover_method_params)

    flatten = itertools.chain.from_iterable

    # TODO(tobi, nacho): pc que sea por couple, no para todas las couples
    def crossover(couples: Couples):
        if pc > 0 and random.random() < pc:  # Children equals parents
            return [child for couple in couples for child in couple]

        return list(flatten(map(lambda couple: child_creation(couple, parent_seq_gen), couples)))

    return crossover


def child_creation(couple: Couple, parent_seq_gen: Callable[[], Iterator[int]]) -> Children:
    children_genes: Tuple[Dict[str, Union[float, Item]], Dict[str, Union[float, Item]]] = ({}, {})

    parent_sequence: Iterator[int] = parent_seq_gen()

    for gene in Character.gene_list:
        parent = next(parent_sequence)
        children_genes[0][gene] = couple[parent].get_gene_by_name(gene)
        children_genes[1][gene] = couple[1 - parent].get_gene_by_name(gene)

    children_item_sets: Tuple[ItemSet, ItemSet] = \
        (_item_set_from_gene_map(children_genes[0]), _item_set_from_gene_map(children_genes[1]))

    children_type: CharacterType = couple[0].type

    assert isinstance(children_genes[0]['height'], float)
    assert isinstance(children_genes[1]['height'], float)
    return [Character(children_type, float(children_genes[0]['height']), children_item_sets[0]),
            Character(children_type, float(children_genes[1]['height']), children_item_sets[1])]


def _item_set_from_gene_map(gene_map: Dict[str, Union[float, Item]]) -> ItemSet:
    assert isinstance(gene_map['weapon'], Item)
    assert isinstance(gene_map['boots'], Item)
    assert isinstance(gene_map['helmet'], Item)
    assert isinstance(gene_map['gauntlets'], Item)
    assert isinstance(gene_map['chest_piece'], Item)
    return ItemSet(gene_map['weapon'], gene_map['boots'], gene_map['helmet'],
                   gene_map['gauntlets'], gene_map['chest_piece'])


def get_k_point_parent_seq(gene_count: int, k: int) -> Iterator[int]:
    points = np.sort(np.array(random.sample(range(gene_count + 1), k)))  # [0, gene_count + 1)
    return map(lambda i: np.searchsorted(points, i, side='right') % 2, range(gene_count))


def get_single_point_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 1)


def get_two_point_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 2)


def get_two_point_parents_annular_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    p1: int = random.randint(0, gene_count - 1)
    l: int = random.randint(0, gene_count - 1)
    p2: int = (p1 + l) % gene_count
    points = np.sort(np.array([p1, p2]))
    return map(lambda i: np.searchsorted(points, i, side='right') % 2, range(gene_count))


uniform_parent_seq_param_schema: ParamValidator = Schema({
    Optional('weight', default=0.5): And(float, lambda weight: 0 <= weight <= 1),
}, ignore_extra_keys=True)


def uniform_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    weight: float = seq_params['weight']
    return map(lambda _: 1 if weight > random.randint(0, 1) else 0, range(gene_count))


_parent_seq_dict: Dict[str, Tuple[ParentSeqGenerator, ParamValidator]] = {
    'single_point': (get_single_point_parent_seq, None),
    'two_point': (get_two_point_parent_seq, None),
    'annular': (get_two_point_parents_annular_seq, None),
    'uniform': (uniform_parent_seq, uniform_parent_seq_param_schema),
}
