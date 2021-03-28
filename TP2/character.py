from enum import Enum
from math import tanh
from typing import Optional, Dict, Callable, List, Union

import numpy as np

from TP2.items import ItemSet, ItemType, Item, ItemAttribute


class CharacterType(Enum):
    warrior = 'warrior'
    archer = 'archer'
    defender = 'defender'
    rogue = 'rogue'


class Character:

    @staticmethod
    def generate_random_height() -> float:
        return np.random.uniform(1.3, 2)

    gene_list: List[str] = sorted(list(map(lambda item_type: item_type.value, ItemType)) + ['height'])

    attr_list: List[str] = sorted(list(map(lambda item_attr: item_attr.value, ItemAttribute)) + ['height'])

    _character_type_fitness_calculator_dict: Dict[CharacterType, Callable[[float, float], float]] = {
        CharacterType.warrior: lambda attack, defence: 0.6 * attack + 0.6 * defence,
        CharacterType.archer: lambda attack, defence: 0.9 * attack + 0.1 * defence,
        CharacterType.defender: lambda attack, defence: 0.3 * attack + 0.8 * defence,
        CharacterType.rogue: lambda attack, defence: 0.8 * attack + 0.3 * defence,
    }

    def __init__(self, character_type: CharacterType, height: float, items: ItemSet) -> None:
        self.type = character_type
        self.height: float = height
        self.items: ItemSet = items

        # Caches para no recalcular propiedades costosas, y mejorar performance
        self._fitness_cache: Optional[float] = None
        self._strength_cache: Optional[float] = None
        self._agility_cache: Optional[float] = None
        self._experience_cache: Optional[float] = None
        self._endurance_cache: Optional[float] = None
        self._vitality_cache: Optional[float] = None

    def get_gene_by_name(self, gene: str) -> Union[Item, float]:
        if gene == 'height':
            return self.height
        else:
            return self.items.get_item(ItemType(gene))

    def get_atm(self) -> float:
        return 0.7 - (3*self.height - 5)**4 + (3*self.height - 5)**2 + self.height/4

    def get_dem(self) -> float:
        return 1.9 + (2.5*self.height - 4.16)**4 - (2.5*self.height - 4.16)**2 - 3*self.height/10

    def _calculate_strength(self) -> float:
        return 100*tanh(0.01*self.items.get_total_strength())

    def _calculate_agility(self) -> float:
        return tanh(0.01*self.items.get_total_agility())

    def _calculate_experience(self) -> float:
        return 0.6*tanh(0.01*self.items.get_total_experience())

    def _calculate_endurance(self) -> float:
        return tanh(0.01*self.items.get_total_endurance())

    def _calculate_vitality(self) -> float:
        return 100*tanh(0.01*self.items.get_total_vitality())

    def get_strength(self) -> float:
        if not self._strength_cache:
            self._strength_cache = self._calculate_strength()
        return self._strength_cache

    def get_agility(self) -> float:
        if not self._agility_cache:
            self._agility_cache = self._calculate_agility()
        return self._agility_cache

    def get_experience(self) -> float:
        if not self._experience_cache:
            self._experience_cache = self._calculate_experience()
        return self._experience_cache

    def get_endurance(self) -> float:
        if not self._endurance_cache:
            self._endurance_cache = self._calculate_endurance()
        return self._endurance_cache

    def get_vitality(self) -> float:
        if not self._vitality_cache:
            self._vitality_cache = self._calculate_vitality()
        return self._vitality_cache

    def get_attack(self) -> float:
        return self._calculate_strength() * (self._calculate_agility() + self._calculate_experience()) * self.get_atm()

    def get_defence(self) -> float:
        return self._calculate_vitality() * (self._calculate_endurance() + self._calculate_experience()) * self.get_dem()

    def _calculate_fitness(self) -> float:
        return Character._character_type_fitness_calculator_dict[self.type](self.get_attack(), self.get_defence())

    def get_fitness(self) -> float:
        if not self._fitness_cache:
            self._fitness_cache = self._calculate_fitness()
        return self._fitness_cache
