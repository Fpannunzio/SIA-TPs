from collections import Collection

import numpy as np

from TP2.character import CharacterType, Character
from TP2.config_loader import Config
from TP2.items import ItemRepositories


class Engine:

    @staticmethod
    def generate_random_height() -> float:
        return np.random.Generator.uniform(1.3, 2)

    def __init__(self, config: Config, item_repositories: ItemRepositories) -> None:
        self.item_repositories = item_repositories
        self.population_type: CharacterType = CharacterType(config.character_type)  # TODO: handle cast error
        self.population_size: int = config.population_size

    def generate_base_population(self) -> Collection[Character]:
        population_size: int = self.population_size
        population_type: CharacterType = CharacterType(self.population_type)  # TODO: handle cast error

        return np.array(
            [Character(population_type, Engine.generate_random_height(), self.item_repositories.generate_random_set())
             for i in range(population_size)])
