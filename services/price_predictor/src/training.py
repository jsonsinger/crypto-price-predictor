from typing import Optional

from loguru import logger

from comet_ml import Experiment

from src.hopsworks_wrapper import HopsworksWrapper
from src.config import config, hopsworks_config, HopsworksConfig, comet_config, CometConfig
from src.models.current_price_baseline import CurrentPriceBaseline

# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def train_model(
    comet_config: CometConfig,
    hopsworks_config: HopsworksConfig,
    feature_view_name: str,
    feature_view_version: int,
    feature_group_name: str,
    feature_group_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days: int,
    forecast_window_min: int,
    test_data_split_size: Optional[float] = 0.2,
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

    # Initialize Comet experiment
    experiment = Experiment(
        api_key=comet_config.comet_api_key,
        project_name=comet_config.comet_project_name,
    )

    # Load features from the Feature Store
    hopsworks_wrapper = HopsworksWrapper(
        ohlc_window_sec=ohlc_window_sec,
        hopsworks_config=hopsworks_config,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
    )

    ohlc_data = hopsworks_wrapper.read_from_offline_store(
        product_id=product_id,
        last_n_days=last_n_days,
    )
    logger.debug(f"Read {len(ohlc_data)} rows of data from the offline Feature Store")
    experiment.log_parameter("ohlc_data_rows", len(ohlc_data))

    # Split the data using scikit-learn train_test_split
    train_df, test_df = train_test_split(ohlc_data, test_size=test_data_split_size, random_state=42)

    # Log the training and testing set sizes with percentages
    logger.debug(f"Training set: {len(train_df)}, {len(train_df) / len(ohlc_data)}%")
    logger.debug(f"Testing set: {len(test_df)}, {len(test_df) / len(ohlc_data)}%")
    
    experiment.log_parameter("train_data_rows", len(train_df))
    experiment.log_parameters("test_data_rows", len(test_df))

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
    
    experiment.log_parameters("X_train_shape", X_train.shape)
    experiment.log_parameters("y_train_shape", y_train.shape)
    experiment.log_parameters("X_test_shape", X_test.shape)
    experiment.log_parameters("y_test_shape", y_test.shape)

    # breakpoint()

    # Build a predictive model
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    logger.debug("Model has been trained")

    # Evaluate the baseline model
    baseline_predictions = model.predict(X_test) # Assuming 'close' is the last known price
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    percentage_error = baseline_mae / y_test.mean() * 100

    experiment.log_metric("Baseline MAE", baseline_mae)
    logger.info(f"Baseline MAE: {baseline_mae}")

    experiment.log_metric("Mean Absolute Percentage Error", percentage_error)   
    logger.info(f"Mean Absolute Percentage Error: {percentage_error:.2f}%")

    experiment.log_metric("Average Target Price", y_test.mean())
    logger.info(f"Average Target Price: {y_test.mean()}")

    # Save the model to the Model Registry

    experiment.end()
    logger.info("Experiment logged to Comet ML successfully!")


if __name__ == "__main__":
    train_model(
        comet_config=comet_config,
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
