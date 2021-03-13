from typing import Optional, Any

from config_loader import Config, StrategyParams


class StrategyStats:

    @staticmethod
    def _validate_property(prop: Optional[Any], error_str: str) -> None:
        if prop is None:
            raise RuntimeError(error_str)

    def __init__(self, config: Config) -> None:

        # Identifiers
        self.strategy_name: str = config.strategy
        self.level_name: str = config.level
        self.strategy_params: StrategyParams = config.strategy_params
        self.render: bool = config.render

        # Properties
        self.runtime: Optional[float] = None
        self.move_count: Optional[int] = None
        self.exploded_node_count: Optional[int] = None
        self.leaf_node_count: Optional[int] = None

    def print_stats(self) -> None:
        self.validate()
        print('-' * 50)
        print(self)
        print('-' * 50)
        if not self.render:
            print('Solution was not rendered because render config was set to false')

    def validate(self) -> None:
        StrategyStats._validate_property(self.runtime, '"Runtime" statistic was not defined')
        StrategyStats._validate_property(self.move_count, '"Move count" statistic was not defined')
        StrategyStats._validate_property(self.exploded_node_count, '"Exploded node count" statistic was not defined')
        StrategyStats._validate_property(self.leaf_node_count, '"Leaf node count" statistic was not defined')

    def __str__(self) -> str:
        return f'Statistics for strategy {self.strategy_name} with params {self.strategy_params} in level "{self.level_name}"\n\n' \
               f'Total Runtime: {self.runtime} seconds\n' \
               f'Total moves (solution depth = total cost): {self.move_count}\n' \
               f'Total exploded nodes: {self.exploded_node_count}\n' \
               f'Total leaf nodes: {self.leaf_node_count}'

    def __repr__(self) -> str:
        return f'StrategyStats(strategy_name={repr(self.strategy_name)}, level_name={repr(self.level_name)}, ' \
               f'runtime={repr(self.runtime)}, move_count={repr(self.move_count)}, ' \
               f'exploded_node_count={repr(self.exploded_node_count)}, leaf_node_count={repr(self.leaf_node_count)})'

    def set_runtime(self, start: float, end: float) -> None:
        self.runtime = end - start

    def set_solution_move_count(self, move_count: int) -> None:
        self.move_count = move_count

    def inc_exploded_node_count(self) -> None:
        self.exploded_node_count = (self.exploded_node_count + 1 if self.exploded_node_count else 1)

    def dec_leaf_node_count(self) -> None:
        if not self.leaf_node_count or self.leaf_node_count == 0:
            raise RuntimeError(f'Leaf count cannot be -1')

        self.leaf_node_count = (self.leaf_node_count + 1 if self.leaf_node_count else 1)

    def inc_leaf_node_count(self) -> None:
        self.leaf_node_count = (self.leaf_node_count + 1 if self.leaf_node_count else 1)
