from typing import List
from time import sleep


class KrakenWebsocketAPI:
    """
    A class for reading real-time trades from the Kraken Websocket API
    """

    def __init__(self, product_id: str):
        """
        Initializes the KrakenWebsocketApi instance

        Args:
            product_id (str): The product ID to get trades for
        """
        self.product_id = product_id

    def get_trades(self) -> List[dict]:
        """
        Get the latest batch of trades for the given product ID

        Returns:
            List[dict]: A list of trade events
        """
        events = [
            {
                "product_id": "ETH/USD",
                "price": "1000",
                "qty": "1",
                "timestamp_ms": "1612345678000",
            }
        ]

        sleep(1)
        return events

    def is_done(self) -> bool:
        """
        Check if the Kraken Websocket API has no more trades to read

        Returns:
            bool: True if there are no more trades to read, False otherwise
        """
        return False
