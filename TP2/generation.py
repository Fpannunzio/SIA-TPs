from typing import List

import numpy as np

from TP2.character import Character, CharacterType
from TP2.items import ItemRepositories

Population = List[Character]


class Generation:

    @staticmethod
    def create_first_generation(generation_size: int, population_type: CharacterType, item_repositories: ItemRepositories) -> 'Generation':
        return Generation([
            Character(population_type, Character.generate_random_height(), item_repositories.generate_random_set())
            for _ in range(generation_size)
        ], 0)

    def __init__(self, population: Population, gen_count: int) -> None:
        self.population: Population = population
        self.gen_count: int = gen_count

    def create_next_generation(self, new_population: Population):
        return Generation(new_population, self.gen_count + 1)

    def get_max_fitness(self) -> float:
        return np.fromiter(map(Character.get_fitness, self.population), np.dtype(float)).max()

    def __len__(self) -> int:
        return len(self.population)