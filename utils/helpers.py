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

    def blob(self) -> str:
        return str(self.value[0])

    @classmethod
    def to_list(cls) -> List[Any]:
        return list(map(lambda c: c, cls))

    @classmethod
    def to_string_list(cls) -> str:
        return ", ".join([str(c) for c in cls])
