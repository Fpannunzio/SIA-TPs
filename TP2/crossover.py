import itertools
import random
from typing import Callable, Collection, Tuple, List, Dict, Union, Iterator

import numpy as np
from schema import Schema, And, Optional, Or

from TP2.character import Character, CharacterType
from TP2.config import Config, Param, ParamValidator
from TP2.items import Item, ItemSet

Crossover = Callable[[Collection[Tuple[Character, Character]]], Collection[Character]]
ParentSeqGenerator = Callable[[int, Param], Iterator[int]]


def _extract_crossover_params(config: Config) -> Param:
    return Config.validate_param(config.crossover, Schema({
        Optional('children_eq_parents_prob', default=0): And(float, lambda p: 0 <= p <= 1),
        Optional('children_per_couple', default=2): And(int, lambda count: count > 0),
        'method': {
            'name': And(str, Or(*tuple(_parent_seq_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_crossover(config: Config) -> Crossover:
    crossover_params: Param = _extract_crossover_params(config)

    method, crossover_method_param_schema = _parent_seq_dict[crossover_params['method']['name']]
    crossover_method_params: Param = crossover_params['method']['params']
    if crossover_method_param_schema:
        crossover_method_params = Config.validate_param(crossover_method_params, crossover_method_param_schema)
    # TODO(tobi): Ver que onda children_per_couple. Como implementarlo
    children_per_couple: int = crossover_params['children_per_couple']
    pc: int = crossover_params['children_eq_parents_prob']

    def parent_seq_gen(): return method(len(Character.gene_list), crossover_method_params)

    flatten = itertools.chain.from_iterable

    def crossover(parents: Collection[Tuple[Character, Character]]):
        if pc > 0 and random.random() < pc:  # Children equals parents
            return [child for couple in parents for child in couple]

        return list(flatten(map(lambda couple: child_creation(couple, parent_seq_gen), parents)))

    return crossover


# TODO(tobi): Que me lo expliquen despacito
def child_creation(couple: Tuple[Character, Character], parent_seq_gen: Callable[[], Iterator[int]]) -> List[Character]:
    children_genes: Tuple[Dict[str, Union[float, Item]], Dict[str, Union[float, Item]]] = {}, {}

    parent_sequence: Iterator[int] = parent_seq_gen()

    for gene in Character.gene_list:
        parent = next(parent_sequence)
        children_genes[0][gene] = couple[parent].get_gene_by_name(gene)
        children_genes[1][gene] = couple[1 - parent].get_gene_by_name(gene)

    children_item_sets: Tuple[ItemSet, ItemSet] = \
        _item_set_from_gene_map(children_genes[0]), _item_set_from_gene_map(children_genes[1])

    children_type: CharacterType = couple[0].type

    return [Character(children_type, children_genes[0]['height'], children_item_sets[0]),
            Character(children_type, children_genes[1]['height'], children_item_sets[1])]


def _item_set_from_gene_map(gene_map: Dict[str, Union[float, Item]]) -> ItemSet:
    return ItemSet(gene_map['weapon'], gene_map['boots'], gene_map['helmet'],
                   gene_map['gauntlets'], gene_map['chest_piece'])


def get_k_point_parent_seq(gene_count: int, k: int) -> Iterator[int]:
    points = np.sort(np.array(random.sample(range(gene_count + 1), k)))  # [0, gene_count + 1)
    return map(lambda i: np.searchsorted(points, i, side='right') % 2, range(gene_count))


def get_single_point_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 1)


def get_two_point_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    return get_k_point_parent_seq(gene_count, 2)


# TODO(tobi): Chequear - Para mi annular es asi
def get_two_point_parents_annular_seq(gene_count: int, seq_params: Param):
    p_and_l: List[int] = random.sample(range(gene_count), 2)
    p1: int = p_and_l[0]
    p2: int = (p1 + p_and_l[1]) % gene_count
    points = np.sort(np.array([p1, p2]))
    return map(lambda i: np.searchsorted(points, i, side='right') % 2, range(gene_count))


uniform_parent_seq_param_schema: ParamValidator = Schema({
    Optional('weight', default=0.5): And(float, lambda weight: 0 <= weight <= 1),
}, ignore_extra_keys=True)


# TODO(tobi): Para mi uniform es asi
def uniform_parent_seq(gene_count: int, seq_params: Param) -> Iterator[int]:
    weight: float = seq_params['weight']
    return map(lambda _: 1 if weight > random.randint(0, 1) else 0, range(gene_count))


_parent_seq_dict: Dict[str, Tuple[ParentSeqGenerator, ParamValidator]] = {
    'single_point': (get_single_point_parent_seq, None),
    'two_point': (get_two_point_parent_seq, None),
    'annular': (get_two_point_parents_annular_seq, None),
    'uniform': (uniform_parent_seq, uniform_parent_seq_param_schema),
}
