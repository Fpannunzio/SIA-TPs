from collections import Collection
from typing import Tuple, Dict, Any, List

from TP2.character import CharacterType, Character, Generation
from TP2.config_loader import Config
from TP2.crossover import Crossover, get_crossover_impl
from TP2.items import ItemRepositories
from TP2.mutation import Mutation, get_mutation_impl
from TP2.couple_selection import ParentSelection, get_couple_selection_impl
from TP2.selection import get_parent_selection, get_survivor_selection, Selection, ParentSelector, SurvivorSelector
from TP2.recombination import Recombination, get_recombination_impl


class Engine:

    def __init__(self, config: Config, item_repositories: ItemRepositories) -> None:
        self.item_repositories = item_repositories
        self.population_type: CharacterType = CharacterType[config.character_class]  # TODO: handle cast error
        self.population_size: int = config.gen_size
        self.config = config

    def generate_base_population(self) -> List[Character]:
        population_size: int = self.population_size
        population_type: CharacterType = CharacterType(self.population_type)  # TODO: handle cast error

        return [Character(population_type, Character.generate_random_height(), self.item_repositories.generate_random_set())
             for i in range(population_size)]

    def resolve_simulation(self) -> Collection[Character]:
        current_gen: Generation = Generation(self.generate_base_population(), 0)

        parent_selection: ParentSelector = get_parent_selection(self.config)
        couple_selection: CoupleSelection = get_couple_selection_impl(self.config)
        crossover: Crossover = get_crossover_impl(self.config)
        mutation: Mutation = get_mutation_impl(self.config)
        recombination: Recombination = get_recombination_impl(self.config)
        survivor_selection: SurvivorSelector = get_survivor_selection(self.config)

        mutation_params: Dict[str, Any] = {
            'probability': 0.5,
        }

        # #TODO real condition
        # condition = True

        for i in range(5):

            parents: Collection[Character] = parent_selection(current_gen)

            parents_couples: Collection[Tuple[Character, Character]] = couple_selection(parents, self.config.k)

            children: Collection[Character] = crossover(parents_couples)

            # TODO sacar mutation params de config
            mutation(children, self.item_repositories, mutation_params)

            current_gen.characters = recombination(current_gen.characters, list(children), current_gen.generation_number, survivor_selection)
            current_gen.generation_number += 1

        return current_gen.characters
