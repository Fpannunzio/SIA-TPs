import math
from typing import Callable, Collection, List, Dict, NamedTuple, Any, Tuple

import numpy as np

from TP2.character import Character
from TP2.config_loader import Config
import random

from TP2.generation import Generation

ParentSelector = Callable[[Generation], Collection[Character]]
SurvivorSelector = Callable[[Generation, int], List[Character]]

SelectionParam = Dict[str, Any]
SelectionDescriptor = NamedTuple('SelectionDescriptor', [('name', str), ('params', SelectionParam)])

Selection = Callable[[Generation, int, SelectionParam], List[Character]]


def get_parent_selection(config: Config) -> ParentSelector:
    # TODO sacar del config nombre del metodo y sus parametros
    first_selection_method: Selection = selection_impl_dict['elite_selection']
    first_method_params: SelectionParam = {}
    second_selection_method: Selection = selection_impl_dict['boltzmann_selection']
    second_method_params: SelectionParam = {
        'initial_temp': 24,
        'final_temp': 12,
        'roulette_method': 'random',
        'k': config.k
    }
    a_value: float = 0.7  # config.a_value

    return lambda generation: first_selection_method(generation, math.ceil(a_value * config.k),
        first_method_params) + second_selection_method(
        generation, math.floor((1 - a_value) * config.k), second_method_params)


def get_survivor_selection(config: Config) -> Selection:
    # TODO sacarlo del config
    first_selection_method: Selection = selection_impl_dict['elite_selection']
    first_method_params: SelectionParam = {}
    second_selection_method: Selection = selection_impl_dict['boltzmann_selection']
    second_method_params: SelectionParam = {
        'initial_temp': 24,
        'final_temp': 12,
        'roulette_method': 'universal',
        'k': config.k
    }
    b_value: float = 0.4  # config.b_value
    return lambda generation, size: first_selection_method(generation, math.ceil(b_value * size),
                                    first_method_params) + second_selection_method(
                                    generation, math.floor((1 - b_value) * size), second_method_params)


def generate_random_numbers(amount: int) -> Collection[float]:
    return [random.random() for _ in range(amount)]


def generate_universal_random_numbers(amount: int) -> Collection[float]:
    r: float = random.random()
    return np.linspace(r / amount, (r + amount - 1) / amount, amount)


# Accumulated sum maintains list order
def calculate_fitness_accumulated_sum(initial_parents: List[Character]) -> Collection[float]:
    fitness_list = np.fromiter(map(Character.get_fitness, initial_parents), float)
    return np.cumsum(fitness_list / fitness_list.sum())


def fitness_and_index(enumeration_tuple: Tuple[int, Character]) -> Tuple[float, int]:

    index, character = enumeration_tuple
    return character.get_fitness(), index


def calculate_ranking_pseudo_fitness_accumulated_sum(initial_parents: List[Character]) -> Collection[float]:
    fitness_list: np.ndarray = np.fromiter(map(fitness_and_index, enumerate(initial_parents)),
                                           np.dtype([('fitness', float), ('index', int)]))

    # Ordenar por fitness de mayor a menor
    fitness_list = np.flipud(np.sort(fitness_list, order='fitness'))

    population_size = np.size(fitness_list)

    # Convertirlo en ranking
    fitness_list['fitness'] = np.linspace(1 - 1 / population_size, 0, population_size)

    # Recuperar por el orden orignal
    fitness_list = np.sort(fitness_list, order='index')['fitness']

    # Acumulada
    return np.cumsum(fitness_list / fitness_list.sum())


def calculate_boltzmann_accumulated_sum(generation: Generation, selection_params: SelectionParam) -> Collection[float]:
    t0: float = selection_params['initial_temp']
    tc: float = selection_params['final_temp']
    k: int = selection_params['k']
    temp: float = tc + (t0 - tc)*math.exp(k*generation.generation*(-1))
    fitness_list = np.fromiter(map(lambda character: math.exp(character.get_fitness() / temp), generation.characters), float)
    mean = np.mean(fitness_list)
    boltzmann_fitness_list = fitness_list / mean
    return np.cumsum(boltzmann_fitness_list / boltzmann_fitness_list.sum())


def elite_selection(generation: Generation, amount: int, selection_params: SelectionParam) -> Collection[Character]:
    return sorted(generation.characters, key=lambda c: c.get_fitness())[:amount]


def random_roulette_selection(generation: Generation, amount, selection_params: SelectionParam) -> List[Character]:
    return roulette_selection(generation.characters, generate_random_numbers(amount),
                              calculate_fitness_accumulated_sum(generation.characters))


def universal_roulette_selection(generation: Generation, amount, selection_params: SelectionParam) -> List[
    Character]:
    return roulette_selection(generation.characters, generate_universal_random_numbers(amount),
                              calculate_fitness_accumulated_sum(generation.characters))


def boltzmann_selection(generation: Generation, amount, selection_params: SelectionParam) -> List[Character]:
    return roulette_selection(generation.characters, roulette_method[selection_params['roulette_method']](amount), calculate_boltzmann_accumulated_sum(generation, selection_params))


def roulette_selection(initial_parents: List[Character], random_numbers: Collection[float],
                       accumulated_sum: Collection[float]) -> List[Character]:
    new_parents: List[Character] = []
    for num in random_numbers:
        new_parents.append(initial_parents[np.searchsorted(accumulated_sum, num)])

    return new_parents


roulette_method: Dict[str, Callable[[int], Collection[float]]] = {
    'random': generate_random_numbers,
    'universal': generate_universal_random_numbers
}

selection_impl_dict: Dict[str, Selection] = {
    'elite_selection': elite_selection,
    'random_roulette_selection': random_roulette_selection,
    'universal_roulette_selection': universal_roulette_selection,
    'boltzmann_selection': boltzmann_selection
}
