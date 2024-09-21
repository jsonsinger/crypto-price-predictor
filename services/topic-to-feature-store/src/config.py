from typing import List

from pydantic_settings import BaseSettings


class TopicToFeatureStoreConfig(BaseSettings):
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_consumer_group: str

    feature_group_name: str
    feature_group_version: int
    feature_group_primary_keys: List[str]
    feature_group_event_time: str

    start_offline_materialization: bool

    class Config:
        env_file = '.env'


class HopsworksConfig(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str

    class Config:
        env_file = 'credentials.env'


fs_config = TopicToFeatureStoreConfig()
hopsworks_config = HopsworksConfig()
