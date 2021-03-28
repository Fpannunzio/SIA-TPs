from typing import List

import numpy as np

from TP2.character import Character, CharacterType
from TP2.items import ItemRepositories


class Generation:

    @staticmethod
    def create_first_generation(generation_size: int, population_type: CharacterType, item_repositories: ItemRepositories) -> 'Generation':
        return Generation([
            Character(population_type, Character.generate_random_height(), item_repositories.generate_random_set())
            for _ in range(generation_size)
        ], 0)

    def __init__(self, population: List[Character], gen_count: int) -> None:
        self.population: List[Character] = population
        self.gen_count: int = gen_count

    def get_max_fitness(self) -> float:
        return np.array(map(Character.get_fitness, self.population)).max()

