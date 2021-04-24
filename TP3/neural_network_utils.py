from typing import Callable, List, TypeVar, Optional

import numpy as np
import attr

from neural_network import NeuralNetwork

NeuralNetworkFactory = Callable[[], NeuralNetwork]
MetricCalculator = Callable[[NeuralNetwork, np.ndarray, np.ndarray], float]

# Generic Internal Variable
_T = TypeVar('_T')


def _assert_not_none(obj: Optional[_T]) -> _T:
    if obj is None:
        raise TypeError()
    return obj


@attr.s(auto_attribs=True)
class CrossValidationResult:
    best_neural_network: NeuralNetwork
    best_error_network: NeuralNetwork
    best_training_points: np.ndarray
    best_training_values: np.ndarray
    best_test_points: np.ndarray
    best_test_values: np.ndarray
    best_error_points: np.ndarray
    best_error_values: np.ndarray
    best_metric: float
    metrics_mean: float
    metrics_std: float


def cross_validation(neural_network_factory: NeuralNetworkFactory,
                     training_points: np.ndarray, training_values: np.ndarray,
                     get_metric: MetricCalculator,
                     test_points_count: int,
                     iteration_count: int) -> CrossValidationResult:
    if (
        test_points_count > len(training_values)//2 or
        test_points_count <= 0 or
        len(training_values) != len(training_points) or
        iteration_count <= 0
    ):
        raise ValueError('Invalid cross validation parameters')
    gt_points: np.ndarray
    gt_values: np.ndarray
    gv_points: np.ndarray
    gv_values: np.ndarray
    be_points: np.ndarray = np.zeros((1, 1))
    be_values: np.ndarray = np.zeros((1, 1))
    current_metric: float
    neural_network: NeuralNetwork
    best_metric: Optional[float] = None
    best_indexes: np.ndarray = np.zeros((1, 1))
    best_neural_network: Optional[NeuralNetwork] = None
    best_error_network: Optional[NeuralNetwork] = None
    all_metrics: List[float] = []

    for _ in range(iteration_count):
        possible_values: np.ndarray = np.arange(len(training_points))

        while len(possible_values) // test_points_count > 0:
            neural_network = neural_network_factory()

            indexes = np.random.choice(possible_values, size=test_points_count, replace=False)
            possible_values = possible_values[~np.isin(possible_values, indexes)]
            gt_points = np.delete(training_points, indexes, axis=0)
            gt_values = np.delete(training_values, indexes, axis=0)
            gv_points = np.take(training_points, indexes, axis=0)
            gv_values = np.take(training_values, indexes, axis=0)

            neural_network.train(gt_points, gt_values)
            current_metric = get_metric(neural_network, gv_points, gv_values)
            all_metrics.append(current_metric)

            if best_error_network is None or neural_network.error < best_error_network.error:
                best_error_network = neural_network
                be_points = gt_points
                be_values = gt_values

            if best_metric is None or best_metric < current_metric or (best_metric == current_metric and neural_network.error < best_neural_network.error):
                best_metric = current_metric
                best_indexes = indexes
                best_neural_network = neural_network

    all_metrics_np: np.ndarray = np.array(all_metrics)
    return CrossValidationResult(
        _assert_not_none(best_neural_network),
        _assert_not_none(best_error_network),
        np.delete(training_points, best_indexes, axis=0),
        np.delete(training_values, best_indexes, axis=0),
        np.take(training_points, best_indexes, axis=0),
        np.take(training_values, best_indexes, axis=0),
        be_points,
        be_values,
        best_metric,
        all_metrics_np.mean(),
        all_metrics_np.std(),
    )
