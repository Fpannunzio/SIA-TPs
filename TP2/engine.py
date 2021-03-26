from collections import Collection
from typing import Tuple, Callable, Dict, Any

import numpy as np

from TP2.character import CharacterType, Character
from TP2.config_loader import Config
from TP2.crossover import Crossover, get_crossover_impl
from TP2.items import ItemRepositories
from TP2.mutation import Mutation, get_mutation_impl
from TP2.couple_selection import ParentSelection, get_couple_selection_impl
from TP2.selection import combinated_selection


class Engine:

    def __init__(self, config: Config, item_repositories: ItemRepositories) -> None:
        self.item_repositories = item_repositories
        self.population_type: CharacterType = CharacterType(config.character_type)  # TODO: handle cast error
        self.population_size: int = config.population_size
        self.config = config

    def generate_base_population(self) -> Collection[Character]:
        population_size: int = self.population_size
        population_type: CharacterType = CharacterType(self.population_type)  # TODO: handle cast error

        return np.array(
            [Character(population_type, Character.generate_random_height(), self.item_repositories.generate_random_set())
             for i in range(population_size)])

    def resolve_simulation(self) -> Collection[Character]:
        current_gen: Collection[Character] = self.generate_base_population()

        parent_selection: Callable[[Collection[Character]], Collection[Character]] = combinated_selection(self.config)
        couple_selection: ParentSelection = get_couple_selection_impl(self.config)
        crossover: Crossover = get_crossover_impl(self.config)
        mutation: Mutation = get_mutation_impl(self.config)

        mutation_params: Dict[str, Any] = {
            'probability': 0.5,
        }

        # #TODO real condition
        # condition = True
        #
        # while condition:

        parents: Collection[Character] = parent_selection(current_gen)

        parents_couples: Collection[Tuple[Character, Character]] = couple_selection(parents)

        children: Collection[Character] = crossover(parents_couples)

        #TODO sacar mutation params de config
        mutation(children, self.item_repositories, mutation_params)

        print(children)

        return []
