from typing import List

import numpy as np

from character import Character, CharacterType
from items import ItemRepositories

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

    def create_next_generation(self, new_population: Population) -> 'Generation':
        return Generation(new_population, self.gen_count + 1)

    def get_max_fitness(self) -> float:
        return np.fromiter(map(Character.get_fitness, self.population), np.dtype(float)).max()

    def get_best_character(self) -> Character:
        return self.population[np.argmax(np.fromiter(map(Character.get_fitness, self.population), np.dtype(float)))]

    def __len__(self) -> int:
        return len(self.population)

    def __repr__(self) -> str:
        return f'Generation=(gen_count={repr(self.gen_count)}, population={repr(self.population)})'

    def __str__(self) -> str:
        return f'Generation Number {self.gen_count}, Size {len(self)}:\n' + \
               '\n'.join(map(lambda char_enum: f'{char_enum[0]}: {char_enum[1]}', enumerate(self.population)))
