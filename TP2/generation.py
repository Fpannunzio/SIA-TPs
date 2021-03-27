from typing import List

from TP2.character import Character, CharacterType
from TP2.items import ItemRepositories


class Generation:

    def __init__(self, characters: List[Character], generation: int) -> None:
        self.characters: List[Character] = characters
        self.generation: int = generation

    @staticmethod
    def generate_first_generation(generation_size: int, character_type: CharacterType, item_repositories: ItemRepositories) -> 'Generation':
        population_type: CharacterType = CharacterType(character_type)  # TODO: handle cast error

        return Generation([Character(population_type, Character.generate_random_height(), item_repositories.generate_random_set())
                for i in range(generation_size)], 0)