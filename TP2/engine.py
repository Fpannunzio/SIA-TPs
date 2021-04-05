from typing import Tuple

from plot import AsyncPlotter
from character import CharacterType, Character
from config import Config
from couple_selection import CouplesSelector, get_couples_selector, Couples
from crossover import Crossover, get_crossover, Children
from end_condition import get_end_condition, AbstractEndCondition
from generation import Generation
from items import ItemRepositories
from mutation import Mutator, get_mutator
from recombination import Recombiner, get_recombiner
from selection import get_parent_selector, get_survivor_selector, ParentSelector, SurvivorSelector, Parents


class Engine:

    def __init__(self, config: Config, item_repositories: ItemRepositories, plotter: AsyncPlotter) -> None:
        self.item_repositories = item_repositories
        self.plotter: AsyncPlotter = plotter

        self.generation_size: int = config.population_size
        self.generation_type: CharacterType = CharacterType[config.character_class]

        self.select_parents: ParentSelector = get_parent_selector(config.parent_selection)
        self.select_couples: CouplesSelector = get_couples_selector(config.parent_coupling)
        self.cross_couples: Crossover = get_crossover(config.crossover)
        self.mutate_children: Mutator = get_mutator(config.mutation)
        self.survivor_selection: SurvivorSelector = get_survivor_selector(config.survivor_selection)
        self.build_new_gen: Recombiner = get_recombiner(config.recombination)
        self.end_condition: AbstractEndCondition = get_end_condition(config.end_condition)

    def resolve_simulation(self) -> Tuple[int, int, Character]:

        current_gen: Generation = Generation.create_first_generation(
            self.generation_size, self.generation_type, self.item_repositories
        )

        best_character: Character = current_gen.get_best_character()
        best_character_gen: int = 0

        self.plotter.start()

        while not self.end_condition.condition_met(current_gen):

            parents: Parents = self.select_parents(current_gen)

            couples: Couples = self.select_couples(parents)

            children: Children = self.cross_couples(couples)

            self.mutate_children(children, self.item_repositories)

            current_gen = self.build_new_gen(current_gen, children, self.survivor_selection)

            self.plotter.publish(current_gen)

            generation_best: Character = current_gen.get_best_character()

            if generation_best.has_higher_fitness(best_character):
                best_character = generation_best
                best_character_gen = current_gen.gen_count

            # print(f'Best from generation {current_gen.gen_count}: {current_gen.get_best_character()}')

        return current_gen.gen_count, best_character_gen, best_character
