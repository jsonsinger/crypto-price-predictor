from datetime import datetime, timezone
from typing import Tuple

from loguru import logger


class DateTimeUtil:
    @staticmethod
    def get_midnight_today(timezone: timezone = timezone.utc) -> datetime:
        """
        Get the timestamp in milliseconds for midnight today

            Returns:
                int: The timestamp in milliseconds for midnight today
        """
        now = datetime.now(timezone)
        midnight = datetime.combine(now.date(), datetime.min.time(), timezone)
        return midnight

    def get_midnight_today_ms(timezone: timezone = timezone.utc) -> datetime:
        """
        Get the timestamp in milliseconds for midnight today

            Returns:
                int: The timestamp in milliseconds for midnight today
        """
        return DateTimeUtil.get_midnight_today().timestamp() * 1000

    @staticmethod
    def calculate_timestamps(
        last_n_days: int, timezone: timezone = timezone.utc
    ) -> Tuple[int, int]:
        """
        Calculate the from_ms and to_ms timestamps for the last `last_n_days` days

            Args:
                last_n_days (int): The number of days to get the timestamps for

            Returns:
                Tuple[int, int]: The from_ms and to_ms timestamps in milliseconds
        """

        # Get the current data at midnight using UTC
        to_ms = int(DateTimeUtil.get_midnight_today_ms(timezone))
        from_ms = to_ms - last_n_days * 24 * 60 * 60 * 1000

        logger.debug(
            f'Calculated timestamps for last {last_n_days} days: from={DateTimeUtil.transform_timestamp_to_datetime(from_ms)}, to={DateTimeUtil.transform_timestamp_to_datetime(to_ms)}'
        )

        return from_ms, to_ms

    @staticmethod
    def transform_timestamp_to_datetime(
        timestamp_ms: int, timezone: timezone = timezone.utc
    ) -> datetime:
        """
        Transform a timestamp in milliseconds to a datatime object
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, timezone)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def transform_nanoseconds_to_datetime(
        timestamp_ns: int, timezone: timezone = timezone.utc
    ) -> datetime:
        """
        Transform a timestamp in nanoseconds to a datetime object
        """
        dt = datetime.fromtimestamp(timestamp_ns / 1e9, timezone)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
