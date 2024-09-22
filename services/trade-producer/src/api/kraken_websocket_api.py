import json
from datetime import datetime, timezone
from typing import List

from loguru import logger
from websocket import create_connection

from .base import KrakenBaseAPI
from .trade import Trade


class KrakenWebsocketAPI(KrakenBaseAPI):
    """
    A class for reading real-time trades from the Kraken Websocket API
    """

    URL = 'wss://ws.kraken.com/v2'

    def __init__(self, product_ids: List[str]):
        """
        Initializes the KrakenWebsocketApi instance

        Args:
            product_ids (List[str]): The product IDs to get trades for
        """

        # Establish a connection to the Kraken Websocket API
        self._ws = create_connection(self.URL)
        logger.debug('Connected to the Kraken Websocket API')

        # Subscribe to the given product IDs
        self._subscribe(product_ids)

    def get_trades(self) -> List[Trade]:
        """
        Get the latest batch of trades for the given product ID

        Returns:
            List[dict]: A list of trade events
        """
        message = self._ws.recv()

        # Retrun an empty list if the message is a heartbeat
        if 'heartbeat' in message:
            logger.debug('Heartbeat received')
            return []

        # Parse the message string into a dictionary
        message = json.loads(message)

        trades = []

        # Extract the trade data from the message
        for trade in message['data']:
            trades.append(
                Trade(
                    product_id=trade['symbol'],
                    price=trade['price'],
                    quantity=trade['qty'],
                    timestamp_ms=self.to_milliseconds(trade['timestamp']),
                )
            )

        return trades

    def is_done(self) -> bool:
        """
        Check if the Kraken Websocket API has no more trades to read

        Returns:
            bool: True if there are no more trades to read, False otherwise
        """
        return False

    def _subscribe(self, product_ids: List[str]):
        """
        Subscribe to trades on the Kraken Websocket API for the given product ID

        Args:
            product_id (str): The product ID of the trade to subscribe to

        Returns:
            None
        """
        logger.debug(f'Subscribing to trades for {product_ids}')

        msg = {
            'method': 'subscribe',
            'params': {'channel': 'trade', 'symbol': product_ids, 'snapshot': False},
        }

        self._ws.send(json.dumps(msg))
        logger.debug(f'Successfully subscribed to trades for {product_ids}')

        # Ignore the initial system status
        _ = self._ws.recv()
        logger.debug(f'Received message: {_}')

        for product_id in product_ids:
            # Ignore the subscription confirmation messages received for each product_id
            _ = self._ws.recv()
            logger.debug(f'Received message: {_}')

    @staticmethod
    def to_milliseconds(timestamp: str) -> int:
        """
        Convert a timestamp string to milliseconds

        Args:
            timestamp (str): The timestamp string

        Returns:
            int: The timestamp in milliseconds
        """

        timestamp = datetime.fromisoformat(timestamp[:-1]).replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1000)
