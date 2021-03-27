import math
from typing import Callable, Collection, List, Dict

import numpy as np

from TP2.character import Character
from TP2.config_loader import Config
import random

Selection = Callable[[List[Character], int], List[Character]]


def combinated_selection(config: Config) -> Callable[[Collection[Character]], Collection[Character]]:
    # TODO sacarlo del config
    first_selection_method: Selection = get_selection_method('elite_selection')
    second_selection_method: Selection = get_selection_method('uniform_roulette_selection')
    a_value: float = 0.7  # config.a_value
    return lambda parents: first_selection_method(parents, math.ceil(a_value * config.gen_size)) + second_selection_method(
        parents, math.floor((1 - a_value) * config.gen_size))


def elite_selection(initial_parents: List[Character], amount: int) -> Collection[Character]:
    sorted(initial_parents, key=lambda c: c.get_fitness())
    return initial_parents[:amount]


def random_roulette_selection(initial_parents: List[Character], amount) -> List[Character]:
    random_numbers: Collection[float] = [random.random() for _ in range(amount)]
    return roulette_selection(initial_parents, random_numbers)


def uniform_roulette_selection(initial_parents: List[Character], amount) -> List[Character]:
    uniform_numbers: Collection[float] = [np.random.uniform(0, 1.0) for _ in range(amount)]
    return roulette_selection(initial_parents, uniform_numbers)


def roulette_selection(initial_parents: List[Character], random_numbers: Collection[float]) -> List[Character]:
    fitness_list = np.fromiter(map(Character.get_fitness, initial_parents), float)
    accumulated_sum = np.cumsum(fitness_list / fitness_list.sum())

    new_parents: List[Character] = []
    for num in random_numbers:
        new_parents.append(initial_parents[np.searchsorted(accumulated_sum, num)])

    return new_parents


def get_selection_method(method: str) -> Selection:
    return selection_impl_dict[method]


selection_impl_dict: Dict[str, Selection] = {
    'elite_selection': elite_selection,
    'random_roulette_selection': random_roulette_selection,
    'uniform_roulette_selection': uniform_roulette_selection
}
