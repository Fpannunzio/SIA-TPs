from config import Param


# TODO(tobi): Matarlo
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

