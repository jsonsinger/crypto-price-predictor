from typing import List

from .base import KrakenBaseAPI
from .trade import Trade


class KrakenRestAPI(KrakenBaseAPI):
    def __init__(self, product_ids: List[str]):
        pass

    def get_trades(self) -> List[Trade]:
        """
        Get the trades from the selected source and return them as a list of Trade objects
        """
        pass

    def is_done(self) -> bool:
        """
        Check if the source has no more trades to provide
        """
        pass
