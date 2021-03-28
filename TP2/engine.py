from typing import Tuple, Collection

from TP2.character import CharacterType, Character
from TP2.config import Config
from TP2.couple_selection import CoupleSelector, get_couple_selector
from TP2.crossover import Crossover, get_crossover
from TP2.end_condition import get_end_condition, AbstractEndCondition
from TP2.generation import Generation
from TP2.items import ItemRepositories
from TP2.mutation import Mutator, get_mutator
from TP2.recombination import Recombiner, get_recombiner
from TP2.selection import get_parent_selector, get_survivor_selector, ParentSelector, SurvivorSelector


class Engine:

    def __init__(self, config: Config, item_repositories: ItemRepositories) -> None:
        self.item_repositories = item_repositories

        self.generation_size: int = config.population_size
        self.generation_type: CharacterType = CharacterType[config.character_class]

        self.parent_selector: ParentSelector = get_parent_selector(config)
        self.couple_selector: CoupleSelector = get_couple_selector(config)
        self.crossover: Crossover = get_crossover(config)
        self.mutation: Mutator = get_mutator(config)
        self.recombination: Recombiner = get_recombiner(config)
        self.survivor_selection: SurvivorSelector = get_survivor_selector(config)
        self.end_condition: AbstractEndCondition = get_end_condition(config)

    def resolve_simulation(self) -> Collection[Character]:

        current_gen: Generation = Generation.create_first_generation(
            self.generation_size, self.generation_type, self.item_repositories
        )

        while not self.end_condition.condition_met(current_gen):

            parents: Collection[Character] = self.parent_selector(current_gen)

            parents_couples: Collection[Tuple[Character, Character]] = self.couple_selector(parents)

            children: Collection[Character] = self.crossover(parents_couples)

            self.mutation(children, self.item_repositories)

            current_gen.population = self.recombination(current_gen, list(children), self.survivor_selection)
            current_gen.gen_count += 1

            print(max(map(Character.get_fitness, current_gen.population)))

        return current_gen.population
