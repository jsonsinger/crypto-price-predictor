from typing import List

from pydantic_settings import BaseSettings


class TradeProducerConfig(BaseSettings):
    kafka_broker_address: str
    kafka_topic: str
    product_ids: List[str]

    # model_config = SettingsConfigDict(env_file=('.env'))

    class Config:
        env_file = '.env'


config = TradeProducerConfig()
