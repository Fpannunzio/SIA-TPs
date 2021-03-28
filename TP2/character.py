from enum import Enum
from math import tanh
from typing import Optional, Dict, Callable, List, Union

import numpy as np

from TP2.items import ItemSet, ItemType, Item


class CharacterType(Enum):
    warrior = 'warrior'
    archer = 'archer'
    defender = 'defender'
    rogue = 'rogue'


character_type_fitness_calculator_dict: Dict[CharacterType, Callable[[float, float], float]] = {
    CharacterType.warrior:  lambda attack, defence: 0.6*attack + 0.6*defence,
    CharacterType.archer:   lambda attack, defence: 0.9*attack + 0.1*defence,
    CharacterType.defender: lambda attack, defence: 0.3*attack + 0.8*defence,
    CharacterType.rogue:    lambda attack, defence: 0.8*attack + 0.3*defence,
}


class Character:

    @staticmethod
    def generate_random_height() -> float:
        return np.random.uniform(1.3, 2)

    @classmethod
    def build_gene_list(cls) -> None:
        ret: List[str] = list(map(lambda item_type: item_type.value, ItemType))
        ret.append('height')
        cls.gene_list = sorted(ret)

    gene_list: List[str]

    def __init__(self, character_type: CharacterType, height: float, items: ItemSet) -> None:
        self.type = character_type
        self.height: float = height
        self.items: ItemSet = items
        self.fitness_cache: Optional[float] = None

    def get_gene_by_name(self, gene: str) -> Union[Item, float]:
        if gene == 'height':
            return self.height
        else:
            return self.items.get_item(ItemType(gene))

    def get_atm(self) -> float:
        return 0.7 - (3*self.height - 5)**4 + (3*self.height - 5)**2 + self.height/4

    def get_dem(self) -> float:
        return 1.9 + (2.5*self.height - 4.16)**4 - (2.5*self.height - 4.16)**2 - 3*self.height/10

    def get_strength(self) -> float:
        return 100*tanh(0.01*self.items.get_total_strength())

    def get_agility(self) -> float:
        return tanh(0.01*self.items.get_total_agility())

    def get_experience(self) -> float:
        return 0.6*tanh(0.01*self.items.get_total_experience())

    def get_endurance(self) -> float:
        return tanh(0.01*self.items.get_total_endurance())

    def get_vitality(self) -> float:
        return 100*tanh(0.01*self.items.get_total_vitality())

    def get_attack(self) -> float:
        return self.get_strength()*(self.get_agility() + self.get_experience()) * self.get_atm()

    def get_defence(self) -> float:
        return self.get_vitality()*(self.get_endurance() + self.get_experience()) * self.get_dem()

    def calculate_fitness(self) -> float:
        return character_type_fitness_calculator_dict[self.type](self.get_attack(), self.get_defence())

    def get_fitness(self) -> float:
        if not self.fitness_cache:
            self.fitness_cache = self.calculate_fitness()

        return self.fitness_cache


# TODO(tobi): Refactor esta cosa rara de python
Character.build_gene_list()
