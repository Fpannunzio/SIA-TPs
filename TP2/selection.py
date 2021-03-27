import math
from typing import Callable, Collection, List, Dict

import numpy as np

from TP2.character import Character
from TP2.config_loader import Config
import random

Selection = Callable[[List[Character], int], List[Character]]
SizedSelection = Callable[[List[Character], int], List[Character]]


def get_parent_selection(config: Config) -> Callable[[Collection[Character]], Collection[Character]]:
    # TODO sacarlo del config
    first_selection_method: Selection = get_selection_method('elite_selection')
    second_selection_method: Selection = get_selection_method('boltzmann_selection')
    a_value: float = 0.7  # config.a_value
    return lambda parents: first_selection_method(parents,
                                                  math.ceil(a_value * config.gen_size)) + second_selection_method(
        parents, math.floor((1 - a_value) * config.gen_size))


def get_survivor_selection(config: Config) -> Selection:
    # TODO sacarlo del config
    first_selection_method: Selection = get_selection_method('elite_selection')
    second_selection_method: Selection = get_selection_method('elite_selection')
    b_value: float = 0.7  # config.b_value
    return lambda parents, size: first_selection_method(parents, math.ceil(b_value * size)) + second_selection_method(
        parents, math.floor((1 - b_value) * size))


def generate_random_numbers(amount: int) -> Collection[float]:
    return [random.random() for _ in range(amount)]


def generate_universal_random_numbers(amount: int) -> Collection[float]:
    r: float = random.random()
    return np.linspace(r / amount, (r + amount - 1) / amount, amount)


def calculate_accumulated_sum(initial_parents: List[Character]) -> Collection[float]:
    fitness_list = np.fromiter(map(Character.get_fitness, initial_parents), float)
    return np.cumsum(fitness_list / fitness_list.sum())


def calculate_ranking_accumulated_sum(initial_parents: List[Character]) -> Collection[float]:
    fitness_list = np.fromiter(map(Character.get_fitness, initial_parents), float)

    return np.cumsum(fitness_list / fitness_list.sum())


def calculate_boltzmann_accumulated_sum(initial_parents: List[Character]) -> Collection[float]:
    temp: int = 10
    fitness_list = np.fromiter(map(lambda character: math.exp(character.get_fitness()/temp), initial_parents), float)
    mean = np.mean(fitness_list)
    boltzmann_fitness_list = fitness_list / mean
    return np.cumsum(boltzmann_fitness_list / boltzmann_fitness_list.sum())


def elite_selection(initial_parents: List[Character], amount: int) -> Collection[Character]:
    sorted(initial_parents, key=lambda c: c.get_fitness())
    return initial_parents[:amount]


def random_roulette_selection(initial_parents: List[Character], amount) -> List[Character]:
    return roulette_selection(initial_parents, generate_random_numbers(amount), calculate_accumulated_sum(initial_parents))


def uniform_roulette_selection(initial_parents: List[Character], amount) -> List[Character]:
    return roulette_selection(initial_parents, generate_universal_random_numbers(amount), calculate_accumulated_sum(initial_parents))


def boltzmann_selection(initial_parents: List[Character], amount) -> List[Character]:
    return roulette_selection(initial_parents, generate_random_numbers(amount), calculate_boltzmann_accumulated_sum(initial_parents))


def roulette_selection(initial_parents: List[Character], random_numbers: Collection[float], accumulated_sum: Collection[float]) -> List[Character]:
    new_parents: List[Character] = []
    for num in random_numbers:
        new_parents.append(initial_parents[np.searchsorted(accumulated_sum, num)])

    return new_parents


def get_selection_method(method: str) -> Selection:
    return selection_impl_dict[method]


selection_impl_dict: Dict[str, Selection] = {
    'elite_selection': elite_selection,
    'random_roulette_selection': random_roulette_selection,
    'uniform_roulette_selection': uniform_roulette_selection,
    'boltzmann_selection': boltzmann_selection,
}
