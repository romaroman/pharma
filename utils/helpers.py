from typing import Optional


class Singleton(type):

    _instance: Optional['Singleton'] = None

    def __call__(cls) -> 'Singleton':
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance
