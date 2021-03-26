from typing import Callable, Collection, List, Dict, Union, Any

from TP2.character import Character
from TP2.config_loader import Config, MutationParams
import random

from TP2.engine import Engine
from TP2.items import Item, ItemRepositories, ItemRepository

Mutation = Callable[[Collection[Character], ItemRepositories, MutationParams], None]

CompleteMutation = Callable[
    [Collection[Character], ItemRepositories, float], None]

gen_setters: Dict[str, Callable[[Character, Union[float, Item]], None]] = {
    'height': lambda character, value: setattr(character, 'height', value),
    'weapon': lambda character, value: setattr(character, 'weapon', value),
    'boots': lambda character, value: setattr(character, 'boots', value),
    'helmets': lambda character, value: setattr(character, 'helmets', value),
    'gauntlets': lambda character, value: setattr(character, 'gauntlets', value),
    'chestpieces': lambda character, value: setattr(character, 'chestpieces', value)
}

items_accessors: Dict[str, Callable[[ItemRepositories], ItemRepository]] = {
    'weapon': lambda items: items.weapons,
    'boots': lambda items: items.boots,
    'helmets': lambda items: items.helmets,
    'gauntlets': lambda items: items.gauntlets,
    'chestpieces': lambda items: items.chestpieces
}

gen_names: List[str] = [
    'height', 'weapon', 'boots', 'helmets', 'gauntlets', 'chestpieces'
]


def get_mutation_impl(config: Config) -> Mutation:
    # TODO por ahora solo esta complete
    return mutation_impl_dict['complete']


def single_gen_mutation(children: Collection[Character], items: ItemRepositories, probability: float, item_type: str):
    for character in children:
        single_gen_mutation_swap(character, items, probability, item_type)


def single_gen_mutation_swap(character: Character, items: ItemRepositories, probability: float, item_type: str):
    if random.random() < probability:
        gen_setters[item_type](character, items_accessors[item_type](items).get_random_item())


def limited_mutation(children: Collection[Character], items: ItemRepositories, probability: float, mutable_genes: int):
    for character in children:
        limited_mutation_swap(character, items, probability, mutable_genes)


def limited_mutation_swap(character, items, probability, mutable_genes):
    random_genes = random.sample(gen_names, mutable_genes)
    multiple_mutation_swap(character, items, probability, random_genes)


def uniform_mutation(children: Collection[Character], items: ItemRepositories, probability: float):
    for character in children:
        multiple_mutation_swap(character, items, probability, gen_names)


def multiple_mutation_swap(character: Character, items: ItemRepositories, probability: float, genes: Collection[str]):
    for gen in genes:
        if gen == 'height' and random.random() < probability:
            gen_setters[gen](character, Engine.generate_random_height())
        else:
            if random.random() < probability:
                gen_setters[gen](character, items_accessors[gen](items).get_random_item())


def complete_mutation(children: Collection[Character], items: ItemRepositories, probability):
    for character in children:
        complete_mutation_swap(character, items, probability)


def complete_mutation_swap(character: Character, items: ItemRepositories, probability: float):
    if random.random() < probability:
        character.height = Engine.generate_random_height()
        character.items = items.generate_random_set()


mutation_impl_dict: Dict[str, Mutation] = {
    'single_gen_mutation': single_gen_mutation,
    'limited_mutation': limited_mutation,
    'uniform_mutation': uniform_mutation,
    'complete': complete_mutation,
}
