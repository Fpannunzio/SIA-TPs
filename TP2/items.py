from enum import Enum
from functools import reduce
from operator import add
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from schema import Schema

import config as conf


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
        self.items: Dict[ItemType, Item] = {
            ItemType.weapon: weapon,
            ItemType.boots: boots,
            ItemType.helmet: helmet,
            ItemType.gauntlets: gauntlets,
            ItemType.chest_piece: chest_piece,
        }

    def get_total_strength(self) -> float:
        return self.sum_items_total('strength')

    def get_total_agility(self) -> float:
        return self.sum_items_total('agility')

    def get_total_experience(self) -> float:
        return self.sum_items_total('experience')

    def get_total_endurance(self) -> float:
        return self.sum_items_total('endurance')

    def get_total_vitality(self) -> float:
        return self.sum_items_total('vitality')

    def sum_items_total(self, attribute: str) -> float:
        return reduce(add, map(lambda item_type: getattr(self.get_item(item_type), attribute), ItemType), 0)

    def get_item(self, item_type: ItemType):
        return self.items[item_type]

    def set_item(self, item_type: ItemType, item: Item):
        self.items[item_type] = item


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

    def get_strength(self, item_id: int) -> float:
        return self.items[item_id][self.strength_pos]

    def get_agility(self, item_id: int) -> float:
        return self.items[item_id][self.agility_pos]

    def get_experience(self, item_id: int) -> float:
        return self.items[item_id][self.experience_pos]

    def get_endurance(self, item_id: int) -> float:
        return self.items[item_id][self.endurance_pos]

    def get_vitality(self, item_id: int) -> float:
        return self.items[item_id][self.vitality_pos]


class ItemRepositories:

    def __init__(self, config: conf.Config) -> None:
        item_files_schema: Dict[str, Any] = dict.fromkeys(map(lambda item_type: item_type.value, ItemType), str)
        item_files: conf.Param = conf.Config.validate_param(config.item_files, Schema(item_files_schema, ignore_extra_keys=True))

        self.repos: Dict[ItemType, ItemRepository] = {
            ItemType.weapon: ItemRepository(item_files[ItemType.weapon.value], ItemType.weapon),
            ItemType.boots: ItemRepository(item_files[ItemType.boots.value], ItemType.boots),
            ItemType.helmet: ItemRepository(item_files[ItemType.helmet.value], ItemType.helmet),
            ItemType.gauntlets: ItemRepository(item_files[ItemType.gauntlets.value], ItemType.gauntlets),
            ItemType.chest_piece: ItemRepository(item_files[ItemType.chest_piece.value], ItemType.chest_piece),
        }

    def generate_random_set(self) -> ItemSet:
        return ItemSet(
            self.get_repo(ItemType.weapon).get_random_item(),
            self.get_repo(ItemType.boots).get_random_item(),
            self.get_repo(ItemType.helmet).get_random_item(),
            self.get_repo(ItemType.gauntlets).get_random_item(),
            self.get_repo(ItemType.chest_piece).get_random_item(),
        )

    def get_repo(self, item_type: ItemType) -> ItemRepository:
        return self.repos[item_type]

    def get_random_item(self, item_type: ItemType) -> Item:
        return self.get_repo(item_type).get_random_item()
