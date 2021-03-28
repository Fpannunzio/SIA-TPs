from collections import Collection
from typing import Tuple, Dict, Any

from TP2.character import CharacterType, Character, Generation
from TP2.config_loader import Config
from TP2.couple_selection import CoupleSelection, get_couple_selection_impl
from TP2.crossover import Crossover, get_crossover_impl
from TP2.end_condition import get_end_condition_impl, AbstractEndCondition
from TP2.generation import Generation
from TP2.items import ItemRepositories
from TP2.mutation import Mutation, get_mutation_impl
from TP2.recombination import Recombination, get_recombination_impl
from TP2.selection import get_parent_selection, get_survivor_selection, ParentSelector, SurvivorSelector


class Engine:

    def __init__(self, config: Config, item_repositories: ItemRepositories) -> None:
        self.item_repositories = item_repositories
        self.generation_type: CharacterType = CharacterType[config.character_class]  # TODO: handle cast error
        self.generation_size: int = config.gen_size
        self.config = config

    def resolve_simulation(self) -> Collection[Character]:
        current_gen: Generation = Generation.generate_first_generation(self.generation_size, self.generation_type,
                                                                       self.item_repositories)

        parent_selection: ParentSelector = get_parent_selection(self.config)
        couple_selection: CoupleSelection = get_couple_selection_impl(self.config)
        crossover: Crossover = get_crossover_impl(self.config)
        mutation: Mutation = get_mutation_impl(self.config)
        recombination: Recombination = get_recombination_impl(self.config)
        survivor_selection: SurvivorSelector = get_survivor_selection(self.config)
        end_condition: AbstractEndCondition = get_end_condition_impl(self.config)

        mutation_params: Dict[str, Any] = {
            'probability': 0.5,
        }

        while end_condition.condition_met(current_gen):
            parents: Collection[Character] = parent_selection(current_gen)

            parents_couples: Collection[Tuple[Character, Character]] = couple_selection(parents, self.config.k)

            children: Collection[Character] = crossover(parents_couples)

            # TODO sacar mutation params de config
            mutation(children, self.item_repositories, mutation_params)

            current_gen.characters = recombination(current_gen, list(children), survivor_selection)
            current_gen.generation += 1

            print(max(map(Character.get_fitness, current_gen.characters)))
        return current_gen.characters
