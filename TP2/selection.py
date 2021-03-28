import math
from typing import Callable, Collection, Dict, Any, Tuple

import numpy as np
from schema import Schema, And
import schema

from TP2.config import Param, ParamValidator
from TP2.character import Character
from TP2.config import Config
import random

from TP2.generation import Generation, Population

# Exported Types
Parents = Population
ParentSelector = Callable[[Generation], Parents]
SurvivorSelector = Callable[[Generation, int], Population]

# Internal Types
Selector = Callable[[Generation, int], Population]
InternalSelector = Callable[[Generation, int, Param], Population]


def _extract_parent_selector_params(config: Config) -> Param:
    method_schema: Dict[Any, Any] = {
        'name': schema.And(str, schema.Or(*tuple(_selector_dict.keys()))),
        schema.Optional('params', default=dict): dict,
    }

    return Config.validate_param(config.parent_selection, Schema({
        'method1': method_schema,
        'method2': method_schema,
        'parent_count': int,
        'weight': And(float, lambda p: 0 <= p <= 1),
    }, ignore_extra_keys=True))


def get_parent_selector(config: Config) -> ParentSelector:
    parent_selector_params: Param = _extract_parent_selector_params(config)

    first_selector_method: Selector = _get_selector(
        parent_selector_params['method1']['name'],
        parent_selector_params['method1']['params']
    )
    second_selector_method: Selector = _get_selector(
        parent_selector_params['method2']['name'],
        parent_selector_params['method2']['params']
    )
    parent_count: int = parent_selector_params['parent_count']
    method_weight: float = parent_selector_params['weight']

    first_method_amount: int = math.ceil(method_weight * parent_count)
    second_method_amount: int = math.floor((1 - method_weight) * parent_count)

    return lambda generation: \
        first_selector_method(generation, first_method_amount) + \
        second_selector_method(generation, second_method_amount)


def _extract_survivor_selector_params(config: Config) -> Param:
    method_schema: Dict[Any, Any] = {
        'name': And(str, schema.Or(*tuple(_selector_dict.keys()))),
        schema.Optional('params', default=dict): dict,
    }

    return Config.validate_param(config.parent_selection, Schema({
        'method1': method_schema,
        'method2': method_schema,
        'weight': And(float, lambda p: 0 <= p <= 1),
    }, ignore_extra_keys=True))


def get_survivor_selector(config: Config) -> SurvivorSelector:
    survivor_selection_params: Param = _extract_survivor_selector_params(config)

    first_selection_method: Selector = _get_selector(
        survivor_selection_params['method1']['name'],
        survivor_selection_params['method1']['params']
    )
    second_selection_method: Selector = _get_selector(
        survivor_selection_params['method2']['name'],
        survivor_selection_params['method2']['params']
    )
    method_weight: float = survivor_selection_params['weight']

    return lambda generation, size: \
        first_selection_method(generation, math.ceil(method_weight * size)) + \
        second_selection_method(generation, math.floor((1 - method_weight) * size))


def _get_selector(method_name: str, params: Param) -> Selector:
    method, selection_param_schema = _selector_dict[method_name]
    validated_params: Param = (Config.validate_param(params, selection_param_schema) if selection_param_schema else params)

    return lambda parents, amount: method(parents, amount, validated_params)


# -------------------------------------- Random Generators -------------------------------------------------------------

def _roulette_random_number_gen(amount: int) -> Collection[float]:
    return [random.random() for _ in range(amount)]


def _universal_random_number_gen(amount: int) -> Collection[float]:
    r: float = random.random()
    return np.linspace(r / amount, (r + amount - 1) / amount, amount)


# Tenias razon Faus, es mejor un mapa
_roulette_method: Dict[str, Callable[[int], Collection[float]]] = {
    'random': _roulette_random_number_gen,
    'universal': _universal_random_number_gen
}


# ------------------------------- Fitness Accumulated Sum Calculators --------------------------------------------------

def _get_accum_sum(probability_array: np.ndarray) -> Collection[float]:
    return np.cumsum(probability_array / probability_array.sum())


# Accumulated sum maintains list order
def _calculate_fitness_accum_sum(population: Population) -> Collection[float]:
    fitness_list = np.fromiter(map(Character.get_fitness, population), float)
    return _get_accum_sum(fitness_list)


def _unpack_fitness_and_index(enum_tuple: Tuple[int, Character]) -> Tuple[float, int]:
    index, character = enum_tuple
    return character.get_fitness(), index


# TODO(tobi): wat, por que recupera el orden original? - No decanto en nada el metodo
def _calculate_ranking_fitness_accum_sum(population: Population) -> Collection[float]:
    fitness_list: np.ndarray = np.fromiter(map(_unpack_fitness_and_index, enumerate(population)),
                                           np.dtype([('fitness', float), ('index', int)]))

    # Ordenar por fitness de mayor a menor
    fitness_list = np.flipud(np.sort(fitness_list, order='fitness'))

    population_size: int = np.size(fitness_list)

    # Convertirlo en ranking
    fitness_list['fitness'] = np.linspace(1 - 1 / population_size, 0, population_size)

    # Recuperar por el orden original
    fitness_list = np.sort(fitness_list, order='index')['fitness']

    # Acumulada
    return _get_accum_sum(fitness_list)


def _calculate_boltzmann_accum_sum(generation: Generation, amount: int, t0: float, tc: float) -> Collection[float]:
    t: float = tc + (t0 - tc) * math.exp(-amount * generation.gen_count)
    fitness_list: np.ndarray = \
        np.fromiter(map(lambda character: math.exp(character.get_fitness() / t), generation.population), float)
    mean = np.mean(fitness_list)
    boltzmann_fitness_list = fitness_list / mean

    return _get_accum_sum(boltzmann_fitness_list)


# -------------------------------------- Selection Strategies ----------------------------------------------------------


# --------------- ELITE ----------------
def elite_selector(generation: Generation, amount: int, selection_params: Param) -> Population:
    return sorted(generation.population, key=lambda c: c.get_fitness())[:amount]


# TODO(tobi): check
def _generic_roulette_selector(population: Population, random_numbers: Collection[float],
                               accumulated_sum: Collection[float]) -> Population:

    return list(map(lambda rand_num_pos: population[rand_num_pos], np.searchsorted(accumulated_sum, random_numbers)))


# ----------------- ROULETTE -------------
def roulette_selector(generation: Generation, amount, selection_params: Param) -> Population:
    return _generic_roulette_selector(
        generation.population,
        _roulette_random_number_gen(amount),
        _calculate_fitness_accum_sum(generation.population)
    )


# ----------------- UNIVERSAL -------------
def universal_selector(generation: Generation, amount, selection_params: Param) -> Population:
    return _generic_roulette_selector(
        generation.population,
        _universal_random_number_gen(amount),
        _calculate_fitness_accum_sum(generation.population)
    )


# ----------------- RANKING -------------
ranking_param_validator: ParamValidator = Schema({
    'roulette_method': _roulette_method.keys()
}, ignore_extra_keys=True)


def ranking_selector(generation: Generation, amount, selection_params: Param) -> Population:
    return _generic_roulette_selector(
        generation.population,
        _roulette_method[selection_params['roulette_method']](amount),
        _calculate_ranking_fitness_accum_sum(generation.population)
    )


# ----------------- BOLTZMANN -------------
# TODO(tobi): No me sale validar que tc < t0
boltzmann_param_validator: ParamValidator = Schema({
    'roulette_method': _roulette_method.keys(),
    'initial_temp': And(float, lambda t0: t0 > 0),
    'final_temp': And(float, lambda tc: 0 < tc)
}, ignore_extra_keys=True)


def boltzmann_selector(generation: Generation, amount, selection_params: Param) -> Population:
    return _generic_roulette_selector(
        generation.population,
        _roulette_method[selection_params['roulette_method']](amount),
        _calculate_boltzmann_accum_sum(
            generation,
            amount,
            selection_params['initial_temp'],
            selection_params['final_temp']
        )
    )


_selector_dict: Dict[str, Tuple[InternalSelector, ParamValidator]] = {
    'elite':        (elite_selector, None),
    'roulette':     (roulette_selector, None),
    'universal':    (universal_selector, None),
    'ranking':      (ranking_selector, ranking_param_validator),
    'boltzmann':    (boltzmann_selector, boltzmann_param_validator)
    # TODO: Torneo 1, torneo 2, verificar ranking
}
