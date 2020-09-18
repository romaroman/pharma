from typing import NoReturn, Dict



class Container:

    def __init__(self) -> NoReturn:
        pass


class Collector:

    class ResultContainer:

        def __init__(self) -> NoReturn:
            self.eval: dict = dict()

    def __init__(self):
        pass
