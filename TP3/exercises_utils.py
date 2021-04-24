import numpy as np
import pandas as pd

from TP3.config import Param


def generate_config() -> Param:
    network_params: Param = {}

    network_params['max_training_iterations'] = 5000
    network_params['weight_reset_threshold'] = network_params['max_training_iterations']
    network_params['max_stale_error_iterations'] = network_params['max_training_iterations']
    network_params['error_goal'] = 0.0000001
    network_params['error_tolerance'] = 0.000000001
    network_params['momentum_factor'] = 0.5
    network_params['base_learning_rate'] = None
    network_params['learning_rate_strategy'] = None

    # Learning Rate Linear Search Params
    network_params['learning_rate_linear_search_params'] = {}
    l_rate_linear_search_params: Param = network_params['learning_rate_linear_search_params']

    l_rate_linear_search_params['max_iterations'] = 1000
    l_rate_linear_search_params['max_value'] = 1
    l_rate_linear_search_params['error_tolerance'] = network_params['error_tolerance']

    # Variable Learning Rate Params
    network_params['variable_learning_rate_params'] = {}
    variable_l_rate_params: Param = network_params['variable_learning_rate_params']

    variable_l_rate_params['down_scaling_factor'] = 0.1
    variable_l_rate_params['up_scaling_factor'] = 0.1# Cuando se use lo setea cada uno
    variable_l_rate_params['positive_trend_threshold'] = 10
    variable_l_rate_params['negative_trend_threshold'] = variable_l_rate_params['positive_trend_threshold'] * 50

    # Network params params
    network_params['network_params'] = {}
    network_params_params: Param = network_params['network_params']
    network_params_params['activation_function'] = 'tanh'
    network_params_params['activation_slope_factor'] = 0.6
    network_params_params['error_function'] = 'quadratic'

    return network_params


def get_training_set(file_name: str, line_count: int, normalize: bool) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None).values
    if normalize:
        training_set = training_set / 100

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set

# https://stackoverflow.com/a/49601444/12270520
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
