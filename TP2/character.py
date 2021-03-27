from enum import Enum
from math import tanh
from random import random
from typing import Optional, Dict, Callable, NamedTuple, List, Collection

import numpy as np

from TP2.items import ItemSet


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

    def __init__(self, character_type: CharacterType, height: float, items: ItemSet) -> None:
        self.type = character_type
        self.height: float = height
        self.items: ItemSet = items
        self.fitness_cache: Optional[float] = None

    def get_atm(self):
        return 0.7 - (3*self.height - 5)**4 + (3*self.height - 5)**2 + self.height/4

    def get_dem(self):
        return 1.9 + (2.5*self.height - 4.16)**4 - (2.5*self.height - 4.16)**2 - 3*self.height/10

    def get_strength(self):
        return 100*tanh(0.01*self.items.get_total_strength())

    def get_agility(self):
        return tanh(0.01*self.items.get_total_agility())

    def get_experience(self):
        return 0.6*tanh(0.01*self.items.get_total_experience())

    def get_endurance(self):
        return tanh(0.01*self.items.get_total_endurance())

    def get_vitality(self):
        return 100*tanh(0.01*self.items.get_total_vitality())

    def get_attack(self):
        return self.get_strength()*(self.get_agility() + self.get_experience()) * self.get_atm()

    def get_defence(self):
        return self.get_vitality()*(self.get_endurance() + self.get_experience()) * self.get_dem()

    def calculate_fitness(self):
        return character_type_fitness_calculator_dict[self.type](self.get_attack(), self.get_defence())

    def get_fitness(self):
        if not self.fitness_cache:
            self.fitness_cache = self.calculate_fitness()

        return self.fitness_cache


Generation = NamedTuple('Generation', [('characters', List[Character]), ('generation_number', int)])
