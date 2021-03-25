from typing import List

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from TP2.config_loader import Config


class Items:

    def __init__(self, config: Config) -> None:
        required_items: List[str] = ['weapons', 'boots', 'helmets', 'gauntlets', 'chestpieces']

        if not all(item_type in config.item_files for item_type in required_items):
            raise ValueError(f'There are arguments missing. Make sure all item types files {required_items} are present')

        self.weapons: ItemRepository = ItemRepository(config.item_files[required_items[0]])
        self.boots: ItemRepository = ItemRepository(config.item_files[required_items[1]])
        self.helmets: ItemRepository = ItemRepository(config.item_files[required_items[2]])
        self.gauntlets: ItemRepository = ItemRepository(config.item_files[required_items[3]])
        self.chestpieces: ItemRepository = ItemRepository(config.item_files[required_items[4]])


class ItemRepository:

    def __init__(self, item_file_path: str) -> None:

        # TODO: na_values is for testing only
        # TODO: Handle tsv not found error
        item_df: DataFrame = pd.read_csv(item_file_path, sep='\t', index_col=0, na_values=20)
        self.items: ndarray = item_df.values

        attrs_list: List[str] = list(item_df.columns.values)

        try:
            self.fuerza_pos: int = attrs_list.index('Fu')
            self.agilidad_pos: int = attrs_list.index('Ag')
            self.pericia_pos: int = attrs_list.index('Ex')
            self.resistencia_pos: int = attrs_list.index('Re')
            self.vida_pos: int = attrs_list.index('Vi')

        except ValueError:
            raise ValueError(f'Items tsv must include Fu, Ag, Ex, Re and Vi headers')

