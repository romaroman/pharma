from enum import Enum
from typing import Optional, List, Any


class Singleton(type):

    _instance: Optional['Singleton'] = None

    def __call__(cls) -> 'Singleton':
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance


class CustomEnum(Enum):

    def __str__(self) -> str:
        return str(self.name)

    @classmethod
    def to_list(cls) -> List[Any]:
        return list(map(lambda c: c, cls))
