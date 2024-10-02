from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class PricePredictorConfig(BaseSettings):
    feature_view_name: str
    feature_view_version: int
    feature_group_name: str
    feature_group_version: int
    ohlc_window_sec: int
    product_id: str
    last_n_days: int
    forecast_window_min: int
    n_search_trials: int
    n_splits: int

    model_config = ConfigDict(
        env_file=".env",
    )


class HopsworksConfig(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str

    model_config = ConfigDict(
        env_file="credentials.env",
        extra = 'allow'
    )
        
class CometConfig(BaseSettings):
    comet_api_key: str
    comet_project_name: str

    model_config = ConfigDict(
        env_file="credentials.env",
        extra = 'allow'
    )


config = PricePredictorConfig()
hopsworks_config = HopsworksConfig()
comet_config = CometConfig()
