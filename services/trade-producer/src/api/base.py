from abc import ABC, abstractmethod
from typing import List

from .trade import Trade


class KrakenBaseAPI(ABC):
    @abstractmethod
    def get_trades(self) -> List[Trade]:
        """
        Get the trades from the selected source and return them as a list of Trade objects
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Check if the source has no more trades to provide
        """
        pass
