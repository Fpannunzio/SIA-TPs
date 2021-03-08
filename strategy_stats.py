from typing import Optional


class StrategyStats:

    @staticmethod
    def _validate_property(prop: Optional, error_str: str) -> None:
        if prop is None:
            raise ValueError(error_str)

    def __init__(self, strategy_name: str, level_name: str) -> None:

        # Identifiers
        self.strategy_name: str = strategy_name
        self.level_name: str = level_name

        # Properties
        self.runtime: Optional[float] = None
        self.move_count: Optional[int] = None
        self.exploded_node_count: Optional[int] = None
        self.max_nodes_stored: Optional[int] = None

    def print_stats(self) -> None:
        self.validate()
        print('-' * 50)
        print(self)

    def validate(self) -> None:
        StrategyStats._validate_property(self.runtime, '"Runtime" statistic was not defined')
        StrategyStats._validate_property(self.move_count, '"Move count" statistic was not defined')
        StrategyStats._validate_property(self.exploded_node_count, '"Exploded node count" statistic was not defined')
        StrategyStats._validate_property(self.max_nodes_stored, '"Max nodes stored" statistic was not defined')

    def __str__(self) -> str:
        return f'Statistics for strategy {self.strategy_name} in level "{self.level_name}"\n\n' \
               f'Total Runtime: {self.runtime} seconds\n' \
               f'Total moves (tree depth): {self.move_count}\n' \
               f'Total exploded nodes: {self.exploded_node_count}\n' \
               f'Max simultaneous nodes stored: {self.max_nodes_stored}\n'

    def __repr__(self) -> str:
        return f'StrategyStats(strategy_name={repr(self.strategy_name)}, level_name={repr(self.level_name)}, ' \
               f'runtime={repr(self.runtime)}, move_count={repr(self.move_count)}, ' \
               f'exploded_node_count={repr(self.exploded_node_count)}, max_nodes_stored={repr(self.max_nodes_stored)})'

    def set_runtime(self, start: float, end: float) -> None:
        self.runtime = end - start

    def set_solution_move_count(self, move_count: int) -> None:
        self.move_count = move_count

    def inc_exploded_node_count(self) -> None:
        self.exploded_node_count = (self.exploded_node_count + 1 if self.exploded_node_count else 1)

    def set_current_nodes_stored(self, current_nodes_stored: int) -> None:
        self.max_nodes_stored = (self.max_nodes_stored if self.max_nodes_stored is not None else 0)
        self.max_nodes_stored = max(self.max_nodes_stored, current_nodes_stored)
