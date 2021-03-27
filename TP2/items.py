from enum import Enum
from functools import reduce
from operator import add
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from TP2.config_loader import Config


class ItemType(Enum):
    weapon = 'weapon'
    boots = 'boots'
    helmet = 'helmet'
    gauntlets = 'gauntlets'
    chest_piece = 'chest_piece'


class Item:

    def __init__(self, item_type: ItemType, strength: float, agility: float,
                 experience: float, endurance: float, vitality: float) -> None:
        self.type: ItemType = item_type
        self.strength = strength
        self.agility = agility
        self.experience = experience
        self.endurance = endurance
        self.vitality = vitality


class ItemSet:

    def __init__(self, weapon: Item, boots: Item, helmet: Item, gauntlets: Item, chest_piece: Item) -> None:
        self.weapon: Item = weapon
        self.boots: Item = boots
        self.helmet: Item = helmet
        self.gauntlets: Item = gauntlets
        self.chest_piece: Item = chest_piece

    def get_total_strength(self):
        return self.sum_items_total('strength')

    def get_total_agility(self):
        return self.sum_items_total('agility')

    def get_total_experience(self):
        return self.sum_items_total('experience')

    def get_total_endurance(self):
        return self.sum_items_total('endurance')

    def get_total_vitality(self):
        return self.sum_items_total('vitality')

    def sum_items_total(self, attribute: str) -> float:
        return reduce(add, map(lambda item_type: getattr(getattr(self, item_type.value), attribute), ItemType), 0)


class ItemRepository:

    def __init__(self, item_file_path: str, item_type: ItemType) -> None:

        # TODO: na_values is for testing only
        # TODO: Handle tsv not found error
        item_df: DataFrame = pd.read_csv(item_file_path, sep='\t', index_col=0, nrows=20)
        self.items: ndarray = item_df.values
        self.type = item_type

        attrs_list: List[str] = list(item_df.columns.values)

        try:
            self.strength_pos: int = attrs_list.index('Fu')
            self.agility_pos: int = attrs_list.index('Ag')
            self.experience_pos: int = attrs_list.index('Ex')
            self.endurance_pos: int = attrs_list.index('Re')
            self.vitality_pos: int = attrs_list.index('Vi')

        except ValueError:
            raise ValueError(f'Items tsv must include Fu, Ag, Ex, Re and Vi headers')

    def get_item(self, item_id: int) -> Item:
        return Item(
            self.type,
            self.get_strength(item_id),
            self.get_agility(item_id),
            self.get_experience(item_id),
            self.get_endurance(item_id),
            self.get_vitality(item_id)
        )

    def get_random_item(self) -> Item:
        return self.get_item(np.random.random_integers(0, np.size(self.items, 0) - 1))

    def get_strength(self, item_id: int):
        return self.items[item_id][self.strength_pos]

    def get_agility(self, item_id: int):
        return self.items[item_id][self.agility_pos]

    def get_experience(self, item_id: int):
        return self.items[item_id][self.experience_pos]

    def get_endurance(self, item_id: int):
        return self.items[item_id][self.endurance_pos]

    def get_vitality(self, item_id: int):
        return self.items[item_id][self.vitality_pos]


class ItemRepositories:

    def __init__(self, config: Config) -> None:
        supported_item_types: List[str] = [item_type.value for item_type in ItemType]
        if not all(item_type in config.item_files for item_type in supported_item_types):
            raise ValueError(
                f'There are arguments missing. Make sure all item types files {supported_item_types} are present')

        self.weapons: ItemRepository = ItemRepository(config.item_files[ItemType.weapon.value], ItemType.weapon)
        self.boots: ItemRepository = ItemRepository(config.item_files[ItemType.boots.value], ItemType.boots)
        self.helmets: ItemRepository = ItemRepository(config.item_files[ItemType.helmet.value], ItemType.helmet)
        self.gauntlets: ItemRepository = ItemRepository(config.item_files[ItemType.gauntlets.value], ItemType.gauntlets)
        self.chest_pieces: ItemRepository = ItemRepository(config.item_files[ItemType.chest_piece.value], ItemType.chest_piece)

    def generate_random_set(self) -> ItemSet:
        return ItemSet(
            self.weapons.get_random_item(),
            self.boots.get_random_item(),
            self.helmets.get_random_item(),
            self.gauntlets.get_random_item(),
            self.chest_pieces.get_random_item()
        )
