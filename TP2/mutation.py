from typing import Callable, Collection, Dict, Tuple

from schema import Schema, Optional, And, Or

from TP2.character import Character
from TP2.config import Config, Param, ParamValidator
import random

from TP2.crossover import Children
from TP2.items import ItemRepositories, ItemType

# Exported Types
Mutator = Callable[[Children, ItemRepositories], None]

# Interna Types
InternalMutator = Callable[[Children, ItemRepositories, float, Param], None]


def _extract_mutator_params(config: Config) -> Param:
    return Config.validate_param(config.mutation, Schema({
        'mutation_probability': And(float, lambda p: 0 <= p <= 1),
        'method': {
            'name': And(str, Or(*tuple(_mutator_dict.keys()))),
            Optional('params', default=dict): dict,
        }
    }, ignore_extra_keys=True))


def get_mutator(config: Config) -> Mutator:
    mutator_params: Param = _extract_mutator_params(config)

    method, mutator_method_param_schema = _mutator_dict[mutator_params['method']['name']]
    mutator_method_params: Param = mutator_params['method']['params']
    if mutator_method_param_schema:
        mutator_method_params = Config.validate_param(mutator_method_params, mutator_method_param_schema)
    mutation_probability: float = mutator_params['mutation_probability']

    return lambda children, item_repo: method(children, item_repo, mutation_probability, mutator_method_params)


# -------------------------------------- Swap Strategies ----------------------------------------------------------

def _single_gen_mutation_swap(child: Character, items_repo: ItemRepositories, probability: float) -> None:
    if random.random() < probability:
        _mutate_child_gene(child, random.choice(Character.gene_list), items_repo)


def _limited_mutation_swap(child: Character, items_repo: ItemRepositories, probability: float, max_mutated_genes_count: int) -> None:
    mutated_genes_count: int = random.randint(1, max_mutated_genes_count)
    random_genes = random.sample(Character.gene_list, mutated_genes_count)
    _multiple_mutation_swap(child, items_repo, probability, random_genes)


def _multiple_mutation_swap(child: Character, items_repo: ItemRepositories, probability: float, genes: Collection[str]) -> None:
    for gene in genes:
        if random.random() < probability:
            _mutate_child_gene(child, gene, items_repo)


def _complete_mutation_swap(child: Character, items_repo: ItemRepositories, probability: float) -> None:
    if random.random() < probability:
        child.height = Character.generate_random_height()
        child.items = items_repo.generate_random_set()


def _mutate_child_gene(child: Character, gene: str, items_repo: ItemRepositories) -> None:
    if gene == 'height':
        child.height = Character.generate_random_height()
    else:  # Es un item
        item_type: ItemType = ItemType(gene)
        child.items.set_item(item_type, items_repo.get_repo(item_type).get_random_item())


# --------------------------------------------- Mutators ---------------------------------------------------------------

def single_gen_mutator(children: Children, items: ItemRepositories, mutation_probability: float, mutation_params: Param) -> None:
    for child in children:
        _single_gen_mutation_swap(child, items, mutation_probability)


limited_mutator_params_schema: ParamValidator = Schema({
    'max_mutated_genes_count': And(int, lambda count: 1 <= count <= len(Character.gene_list))
}, ignore_extra_keys=True)


def limited_mutator(children: Children, items: ItemRepositories, mutation_probability: float, mutation_params: Param) -> None:
    for child in children:
        _limited_mutation_swap(child, items, mutation_probability, mutation_params['max_mutated_genes_count'])


def uniform_mutator(children: Children, items: ItemRepositories, mutation_probability: float, mutation_params: Param) -> None:
    for child in children:
        _multiple_mutation_swap(child, items, mutation_probability, Character.gene_list)


def complete_mutator(children: Children, items: ItemRepositories, mutation_probability: float, mutation_params: Param) -> None:
    for child in children:
        _complete_mutation_swap(child, items, mutation_probability)


_mutator_dict: Dict[str, Tuple[InternalMutator, ParamValidator]] = {
    'single_gen': (single_gen_mutator, None),
    'limited': (limited_mutator, limited_mutator_params_schema),
    'uniform': (uniform_mutator, None),
    'complete': (complete_mutator, None),
}
