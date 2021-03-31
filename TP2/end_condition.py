import time
from abc import ABC, abstractmethod
from typing import Dict, Type, Tuple

import numpy as np
from schema import Schema, And, Optional, Or

from config import Config, Param, ParamValidator
from generation import Generation

MAX_GENERATIONS_ALLOWED: int = 1000


class AbstractEndCondition(ABC):

    @abstractmethod
    def __init__(self, params: Param) -> None:
        pass

    @abstractmethod
    def condition_met(self, generation: Generation) -> bool:
        pass


def _extract_end_condition_params(end_condition_params: Param) -> Param:
    return Config.validate_param(end_condition_params, Schema({
        'name': And(str, Or(*tuple(_end_condition_dict.keys()))),
        Optional('params', default=dict): dict,
    }, ignore_extra_keys=True))


def get_end_condition(end_condition_params: Param) -> AbstractEndCondition:
    end_condition_params = _extract_end_condition_params(end_condition_params)

    end_condition_type, end_condition_method_params_schema = _end_condition_dict[end_condition_params['name']]
    end_condition_method_params = end_condition_params['params']
    if end_condition_method_params_schema:
        end_condition_method_params = Config.validate_param(end_condition_method_params, end_condition_method_params_schema)

    return end_condition_type(end_condition_method_params)


end_by_time_params_schema: ParamValidator = Schema({
    'runtime': And(Or(float, int), lambda runtime: runtime > 0)
}, ignore_extra_keys=True)


class EndByTime(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.has_started: bool = False
        self.runtime = params['runtime']
        self.limit_time: float = -1

    def condition_met(self, generation: Generation) -> bool:
        if not self.has_started:
            self.limit_time = time.perf_counter() + self.runtime

        return time.perf_counter() > self.limit_time


end_by_generation_params_schema: ParamValidator = Schema({
    'limit_generation': And(int, lambda limit: 0 < limit <= MAX_GENERATIONS_ALLOWED)
}, ignore_extra_keys=True)


class EndByGeneration(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.limit_generation: int = params['limit_generation']

    def condition_met(self, generation: Generation) -> bool:
        return generation.gen_count >= self.limit_generation


end_by_fitness_params_schema: ParamValidator = Schema({
    'target_fitness': And(float, lambda fitness: 0 < fitness)
}, ignore_extra_keys=True)


class EndByFitness(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.target_fitness: float = params['target_fitness']

    def condition_met(self, generation: Generation) -> bool:
        return generation.get_max_fitness() >= self.target_fitness


end_by_fitness_convergence_params_schema: ParamValidator = Schema({
    'limit_generation': And(int, lambda limit: 0 < limit <= MAX_GENERATIONS_ALLOWED),
    'number_of_generations': And(int, lambda number: 0 < number <= MAX_GENERATIONS_ALLOWED),
    'epsilon': And(float, lambda epsilon: 0 < epsilon)
}, ignore_extra_keys=True)


class EndByFitnessConvergence(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.epsilon: float = params['epsilon']
        self.number_of_generations: int = params['number_of_generations']
        self.limit_generation: int = params.get('limit_generation', MAX_GENERATIONS_ALLOWED)
        self.previous_fitness_values: np.ndarray = np.zeros(1)

    def condition_met(self, generation: Generation) -> bool:
        self.previous_fitness_values: np.ndarray = np.append(self.previous_fitness_values, generation.get_max_fitness())

        if np.size(self.previous_fitness_values) <= self.number_of_generations:
            return False

        if np.size(self.previous_fitness_values, axis=0) > self.limit_generation + 1:
            return True

        return np.gradient(self.previous_fitness_values[-self.number_of_generations:]).max() < self.epsilon


end_by_diversity_convergence_params_schema: ParamValidator = Schema({
    'limit_generation': And(int, lambda limit: 0 < limit <= MAX_GENERATIONS_ALLOWED),
    'number_of_generations': And(int, lambda number: 0 < number <= MAX_GENERATIONS_ALLOWED),
    'epsilon': And(float, lambda epsilon: 0 < epsilon)
}, ignore_extra_keys=True)


class EndByDiversityConvergence(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.epsilon: float = params['epsilon']
        self.number_of_generations: int = params['number_of_generations']
        self.limit_generation: int = params.get('limit_generation', MAX_GENERATIONS_ALLOWED)
        self.previous_diversity_values: np.ndarray = np.zeros((1, 6))
        self.diversity_index = 0

    def condition_met(self, generation: Generation) -> bool:
        # np.insert(self.previous_diversity_values, self.diversity_index, generation.get_diversity(), axis=0)
        #
        # self.diversity_index += 1

        self.previous_diversity_values: np.ndarray = np.append(self.previous_diversity_values,
                                                               generation.get_diversity().reshape(1, 6), axis=0)

        if np.size(self.previous_diversity_values, axis=0) <= self.number_of_generations:
            return False

        if np.size(self.previous_diversity_values, axis=0) > self.limit_generation + 1:
            return True

        return np.gradient(np.swapaxes(self.previous_diversity_values[-self.number_of_generations:], 0, 1),
                           axis=1).max() < self.epsilon


_end_condition_dict: Dict[str, Tuple[Type[AbstractEndCondition], ParamValidator]] = {
    'by_time': (EndByTime, end_by_time_params_schema),
    'by_generation': (EndByGeneration, end_by_generation_params_schema),
    'by_fitness': (EndByFitness, end_by_fitness_params_schema),
    'by_fitness_convergence': (EndByFitnessConvergence, end_by_fitness_convergence_params_schema),
    'by_diversity_convergence': (EndByDiversityConvergence, end_by_diversity_convergence_params_schema),
}
