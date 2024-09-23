import hashlib
import json
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

from .base import KrakenBaseAPI
from .trade import Trade
from .utils import DateTimeUtil


class KrakenRestAPI(KrakenBaseAPI):
    URL = 'https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}'

    def __init__(
        self,
        product_ids: List[str],
        last_n_days: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialises an instance of the Kraken Rest API

        Args:
            product_ids (List[str]): The product IDs to get trades for
            last_n_days (int): The number of days to get trades for
            cache_dir (Optional[str]): The directory to cache the trades in

        Returns:
            None
        """
        self.product_ids = product_ids
        self.from_ms, self.to_ms = DateTimeUtil.calculate_timestamps(last_n_days)

        logger.debug(
            f'Initialised KrakenRestAPI with product_ids={product_ids}, last_n_days={last_n_days}, from_ms={self.from_ms}, to_ms={self.to_ms}, cache_dir={cache_dir}'
        )

        # The timestamp of the last trade
        self.last_trade_ms = self.from_ms

        # The cache to store historical data to speed up service restarts
        self.use_cache = False
        if cache_dir is not None:
            self.cache = TradeCache(cache_dir)
            self.use_cache = True

    def get_trades(self) -> List[Trade]:
        """
        Get the trades from the selected source and return them as a list of Trade objects
        """

        since_ns = self.last_trade_ms * 1_000_000
        all_trades = []

        for product_id in self.product_ids:
            url = self.URL.format(product_id=product_id, since_sec=since_ns)

            if self.use_cache and self.cache.has(url):
                trades = self._read_trades_from_cache(url=url)
                logger.debug(
                    f'Loaded {len(trades)} trades for {self.product_id}, since={DateTimeUtil.transform_nanoseconds_to_datetime(since_ns)} from the cache'
                )
            else:
                trades = self._fetch_trades_from_api(
                    url=url, product_id=product_id, since_ns=since_ns
                )

            # Update the last trade timestamp
            if trades[-1].timestamp_ms == self.last_trade_ms:
                self.last_trade_ms = trades[-1].timestamp_ms + 1
                logger.debug(
                    f'Updated last_trade_ms by 1 to reduce repeating the api requests {self.last_trade_ms}'
                )
            else:
                self.last_trade_ms = trades[-1].timestamp_ms

            # Filter out trades that are outside the time range
            trades = [trade for trade in trades if trade.timestamp_ms <= self.to_ms]

            # Append the trades to the list of all trades
            all_trades.extend(trades)

        return all_trades

    def is_done(self) -> bool:
        """
        Check if the source has no more trades to provide
        """
        # logger.debug(f'Has more trades to fetch: {self.last_trade_ms < self.to_ms}')
        return self.last_trade_ms >= self.to_ms

    def _read_trades_from_cache(self, url: str) -> List[Trade]:
        """
        Read the trades from the cache

        Args:
            url (str): The URL of the trades

        Returns:
            List[Trade]: The cached trades
        """
        trades = []
        if self.use_cache and self.cache.has(url):
            trades = self.cache.read(url)
        return trades

    def _fetch_trades_from_api(
        self, url: str, product_id: str, since_ns: int
    ) -> Tuple[bool, List[Trade]]:
        """
        Fetch the trades from the Kraken REST API with retries

        Args:
            url (str): The URL to fetch the trades from
            product_id (str): The product ID to fetch the trades for
            since_ns (int): The timestamp in nanoseconds to fetch the trades from

        Returns:
            Tuple[bool, List[Trade]]: A tuple with a boolean indicating if the fetch was successful and the trades
        """
        max_retries = 3
        backoff_factor = 2
        delay = 30  # Initial delay in seconds

        payload = {}
        headers = {}
        logger.debug(f'Fetching trades from {url}')

        for attempt in range(max_retries):
            try:
                # Fetch the trades from the Kraken REST API
                response = requests.request('GET', url, headers=headers, data=payload)

                # Parse the string response into a dictionary
                data = json.loads(response.text)

                if 'error' in data and 'EGeneral:Too many requests' in data['error']:
                    # Slow down the rate at which we are making requests to Kraken's API
                    raise requests.exceptions.RequestException('Rate limit exceeded')

                trades = [
                    Trade(
                        price=float(trade[0]),
                        quantity=float(trade[1]),
                        timestamp_ms=int(trade[2] * 1e3),
                        product_id=product_id,
                    )
                    for trade in data['result'][product_id]
                ]

                logger.debug(
                    f'Fetched {len(trades)} trades for {product_id}, since={DateTimeUtil.transform_nanoseconds_to_datetime(since_ns)} from the Kraken REST API'
                )

                if self.use_cache:
                    self.cache.write(url, trades)
                    logger.debug(
                        f'Cached {len(trades)} trades for {product_id}, since={DateTimeUtil.transform_nanoseconds_to_datetime(since_ns)}'
                    )

                return trades

            except requests.exceptions.RequestException as e:
                logger.error(f'Error fetching trades: {e}')
                if attempt < max_retries - 1:
                    sleep_time = delay * backoff_factor**attempt
                    logger.info(f'Sleeping for {sleep_time} seconds before retrying')
                    sleep(sleep_time)
                else:
                    logger.error('Max retries exceeded')
                    raise e


class TradeCache:
    def __init__(self, cache_dir: str) -> None:
        """
        A class for handling the caching of trades fetch from the Kraken REST API

        Args:
            cache_dir (str): The directory to cache the trades in
        """
        self.cache_dir = cache_dir

        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def read(self, url: str) -> Optional[List[Trade]]:
        """
        Read the trades from the cache

        Args:
            url (str): The URL of the trades

        Returns:
            Optional[List[Trade]]: The cached trades if they exist, None otherwise
        """

        file_path = self._get_file_path(url)

        if not file_path.exists():
            return None

        data = pd.read_parquet(file_path)

        return [Trade(**trade) for trade in data.to_dict(orien='records')]

    def write(self, url: str, trades: List[Trade]) -> None:
        """
        Write the trades to the cache

        Args:
            url (str): The URL of the trades
            trades (List[Trade]): The trades to cache

        Returns:
            None
        """
        if not trades:
            return None

        data = pd.DataFrame([trade.model_dump() for trade in trades])

        file_path = self._get_file_path(url)
        data.to_parquet(file_path)

    def has(self, url: str) -> bool:
        """
        Check if the trades are in the cache

        Args:
            url (str): The URL of the trades

        Returns:
            bool: True if the trades are in the cache, False otherwise
        """
        file_path = self._get_file_path(url)
        return file_path.exists()

    def _get_file_path(self, url: str) -> str:
        """
        Get the file path for the given URL

        Args:
            url (str): The URL of the trades

        Returns:
            str: The file path for the given URL
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f'{url_hash}.parquet'
