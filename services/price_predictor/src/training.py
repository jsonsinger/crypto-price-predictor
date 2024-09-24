from typing import Optional

from loguru import logger

from src.ohlc_data_reader import OhlcDataReader
from src.config import config, hopsworks_config, HopsworksConfig
from src.models.current_price_baseline import CurrentPriceBaseline

# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(
    hopsworks_config: HopsworksConfig,
    feature_view_name: str,
    feature_view_version: int,
    feature_group_name: str,
    feature_group_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days: int,
    forecast_window_min: int,
    test_data_split_size: Optional[float] = 0.3,
):
    """
    Read features from the Feature Store,
    Train a predictive model,
    Save the model to the Model Registry

    Args:
        hopsworks_config (HopsworksConfig): The Hopsworks configuration
        feature_view_name (str): The name of the feature view
        feature_view_version (int): The version of the feature view
        feature_group_name (str): The name of the feature group
        feature_group_version (int): The version of the feature group
        ohlc_window_sec (int): The window size in seconds for the OHLC data
        product_id (str): The product ID
        last_n_days (int): The number of days of OHLC data to read
        forecast_window_min (int): The forecast window size in minutes
        test_data_split_size (float): The size of the test data split (default: 0.3)

    Returns:
        None
    """

    # Load features from the Feature Store
    ohlc_data_reader = OhlcDataReader(
        ohlc_window_sec=ohlc_window_sec,
        hopsworks_config=hopsworks_config,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
    )

    ohlc_data = ohlc_data_reader.read_from_offline_store(
        product_id=product_id,
        last_n_days=last_n_days,
    )
    logger.debug(f"Read {len(ohlc_data)} rows of data from the offline Feature Store")

    # Split the data into training and test sets (70/30 split)
    # TODO Add validation test set and look at train_test_split from scikit-learn

    test_size = int(len(ohlc_data) * test_data_split_size)
    train_df = ohlc_data[:-test_size]
    test_df = ohlc_data[-test_size:]
    logger.debug(f"Training set: {len(train_df)}, {len(train_df) / len(ohlc_data)}%")
    logger.debug(f"Testing set: {len(test_df)}, {len(test_df) / len(ohlc_data)}%")

    # Add a column with the target price we want our model to predict
    train_df["target_price"] = train_df["close"].shift(-forecast_window_min)
    test_df["target_price"] = test_df["close"].shift(-forecast_window_min)
    logger.debug(
        f"Added target price column with forecast window of {forecast_window_min} minutes"
    )

    # Drop rows with NaN values
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Split the data into features and target
    y_train = train_df["target_price"]
    X_train = train_df.drop(columns=["target_price"])
    y_test = test_df["target_price"]
    X_test = test_df.drop(columns=["target_price"])

    # Log the dimensions of the features and targets
    logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # breakpoint()

    # Build a predictive model
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    logger.debug("Model has been trained")

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Mean Absolute Error: {mae}")

    # Save the model to the Model Registry


if __name__ == "__main__":
    train_model(
        hopsworks_config=hopsworks_config,
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        ohlc_window_sec=config.ohlc_window_sec,
        product_id=config.product_id,
        last_n_days=config.last_n_days,
        forecast_window_min=config.forecast_window_min,
    )
