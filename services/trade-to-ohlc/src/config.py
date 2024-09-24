from typing import Optional

from pydantic_settings import BaseSettings


class TradeToOHLCVConfig(BaseSettings):
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str
    ohlc_window: int
    backfill_mode: Optional[bool] = False

    class Config:
        env_file = '.env'


ohlc_config = TradeToOHLCVConfig()
