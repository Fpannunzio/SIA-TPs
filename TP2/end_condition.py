from abc import ABC, abstractmethod
import time
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Type

import numpy as np

from TP2.character import Character, Generation
from TP2.config_loader import Config, Param

DEFAULT_LIMIT_GENERATION: int = 1000


class AbstractEndCondition(ABC):

    @abstractmethod
    def __init__(self, params: Param) -> None:
        pass

    @abstractmethod
    def condition_met(self, generation: Generation) -> bool:
        pass


def get_end_condition_impl(config: Config) -> AbstractEndCondition:
    end_condition_params: Param = config.end_condition

    return end_condition_dict[end_condition_params['name']](end_condition_params)


class EndByTime(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.limit_time: float = time.perf_counter() + params['runtime']

    def condition_met(self, generation: Generation) -> bool:
        return time.perf_counter() > self.limit_time


class EndByGeneration(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.limit_generation: int = params['limit_generation']

    def condition_met(self, generation: Generation) -> bool:
        return generation.generation_number >= self.limit_generation


class EndByFitness(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.target_fitness: int = params['target_fitness']

    def condition_met(self, generation: Generation) -> bool:
        return get_max_fitness(generation.characters) >= self.target_fitness


class EndByFitnessConvergence(AbstractEndCondition):

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.epsilon: float = params['epsilon']
        self.number_of_generations: int = params['number_of_generations']
        self.limit_generation: int = params.get('limit_generation', DEFAULT_LIMIT_GENERATION)
        self.previous_fitness_values: np.ndarray = np.empty(self.limit_generation)

    def condition_met(self, generation: Generation) -> bool:

        np.append(self.previous_fitness_values, get_max_fitness(generation.characters))

        return np.gradient(self.previous_fitness_values[-self.number_of_generations:]).max() < self.epsilon


def get_max_fitness(characters: List[Character]) -> float:
    return np.array(map(Character.get_fitness, characters)).max()

end_condition_dict: Dict[str, Type[AbstractEndCondition]] = {
    'by_time': EndByTime,
    'by_generation': EndByGeneration,
    'by_fitness': EndByFitness,
    'by_fitness_convergence': EndByFitnessConvergence,
}
