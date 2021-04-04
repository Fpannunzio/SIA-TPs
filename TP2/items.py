import random
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


class ItemAttribute(Enum):
    strength = 'strength'
    agility = 'agility'
    experience = 'experience'
    endurance = 'endurance'
    vitality = 'vitality'


class Item:

    def __init__(self, item_type: ItemType, item_id: int, strength: float, agility: float,
                 experience: float, endurance: float, vitality: float) -> None:
        self.type: ItemType = item_type
        self.item_id = item_id
        self.attributes: Dict[ItemAttribute, float] = {
            ItemAttribute.strength: strength,
            ItemAttribute.agility: agility,
            ItemAttribute.experience: experience,
            ItemAttribute.endurance: endurance,
            ItemAttribute.vitality: vitality,
        }

    def get_attribute(self, item_attr: ItemAttribute) -> float:
        return self.attributes[item_attr]

    def full_repr(self):
        return f'Item(type={repr(self.type.value)},' \
               f'id={repr(self.item_id)}, ' \
               f'strength={repr(self.get_attribute(ItemAttribute.strength))}, ' \
               f'agility={repr(self.get_attribute(ItemAttribute.agility))}, ' \
               f'experience={repr(self.get_attribute(ItemAttribute.experience))}, ' \
               f'endurance={repr(self.get_attribute(ItemAttribute.endurance))}, ' \
               f'vitality={repr(self.get_attribute(ItemAttribute.vitality))})'

    def id_repr(self):
        return f'Item(type={repr(self.type.value)}, id={repr(self.item_id)})'

    def __repr__(self) -> str:
        return self.id_repr()
        # return self.full_repr()


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
        return self.sum_items_total(ItemAttribute.strength)

    def get_total_agility(self) -> float:
        return self.sum_items_total(ItemAttribute.agility)

    def get_total_experience(self) -> float:
        return self.sum_items_total(ItemAttribute.experience)

    def get_total_endurance(self) -> float:
        return self.sum_items_total(ItemAttribute.endurance)

    def get_total_vitality(self) -> float:
        return self.sum_items_total(ItemAttribute.vitality)

    def sum_items_total(self, attribute: ItemAttribute) -> float:
        return reduce(add, map(lambda item_type: self.get_item(item_type).get_attribute(attribute), ItemType), 0)

    def get_item(self, item_type: ItemType) -> Item:
        return self.items[item_type]

    def set_item(self, item_type: ItemType, item: Item) -> None:
        self.items[item_type] = item

    def __repr__(self) -> str:
        return f'ItemSet(weapon={repr(self.get_item(ItemType.weapon))}, boots={repr(self.get_item(ItemType.boots))}, ' \
               f'helmet={repr(self.get_item(ItemType.helmet))}, gauntlets={repr(self.get_item(ItemType.gauntlets))}, ' \
               f'chest_piece={repr(self.get_item(ItemType.chest_piece))})'


class ItemRepository:

    attribute_tsv_header_dict: Dict[ItemAttribute, str] = {
        ItemAttribute.strength: 'Fu',
        ItemAttribute.agility: 'Ag',
        ItemAttribute.experience: 'Ex',
        ItemAttribute.endurance: 'Re',
        ItemAttribute.vitality: 'Vi',
    }

    @staticmethod
    def get_attr_tsv_header(item_attr: ItemAttribute) -> str:
        return ItemRepository.attribute_tsv_header_dict[item_attr]

    def __init__(self, item_file_path: str, item_type: ItemType) -> None:

        try:
            item_df: DataFrame = pd.read_csv(item_file_path, sep='\t', index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {item_file_path} configured to load {item_type.value}s was not found')

        self.items: ndarray = item_df.values
        self.type = item_type

        attrs_list: List[str] = list(item_df.columns.values)

        try:
            self.strength_pos: int = attrs_list.index(ItemRepository.get_attr_tsv_header(ItemAttribute.strength))
            self.agility_pos: int = attrs_list.index(ItemRepository.get_attr_tsv_header(ItemAttribute.agility))
            self.experience_pos: int = attrs_list.index(ItemRepository.get_attr_tsv_header(ItemAttribute.experience))
            self.endurance_pos: int = attrs_list.index(ItemRepository.get_attr_tsv_header(ItemAttribute.endurance))
            self.vitality_pos: int = attrs_list.index(ItemRepository.get_attr_tsv_header(ItemAttribute.vitality))

        except ValueError:
            raise ValueError(f'Items tsv must include this headers: {ItemRepository.attribute_tsv_header_dict.values()}')

    def get_item(self, item_id: int) -> Item:
        return Item(
            self.type,
            item_id,
            self.get_strength(item_id),
            self.get_agility(item_id),
            self.get_experience(item_id),
            self.get_endurance(item_id),
            self.get_vitality(item_id)
        )

    def get_random_item(self) -> Item:
        return self.get_item(random.randint(0, np.size(self.items, 0) - 1))

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

    def __init__(self, item_files: conf.Param) -> None:
        item_files_schema: Dict[str, Any] = dict.fromkeys(map(lambda item_type: item_type.value, ItemType), str)
        item_files = conf.Config.validate_param(item_files, Schema(item_files_schema, ignore_extra_keys=True))

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
