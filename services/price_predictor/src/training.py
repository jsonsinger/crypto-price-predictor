from typing import Optional

from loguru import logger

from comet_ml import Experiment

from src.hopsworks_wrapper import HopsworksWrapper
from src.config import config, hopsworks_config, HopsworksConfig, comet_config, CometConfig
from src.models.current_price_baseline import CurrentPriceBaseline
from src.technical_indicators import add_technical_indicators, add_lagged_features
from xgboost import XGBRegressor

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
        comet_config (CometConfig): The Comet configuration
        hopsworks_config (HopsworksConfig): The Hopsworks configuration
        feature_view_name (str): The name of the feature view
        feature_view_version (int): The version of the feature view
        feature_group_name (str): The name of the feature group
        feature_group_version (int): The version of the feature group
        ohlc_window_sec (int): The window size in seconds for the OHLC data
        product_id (str): The product ID
        last_n_days (int): The number of days of OHLC data to read
        forecast_window_min (int): The forecast window size in minutes
        test_data_split_size (float): The size of the test data split (default: 0.2)

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
    experiment.log_parameter("test_data_rows", len(test_df))

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

    # Drop categorical features (for testing purposes)
    X_train = X_train.drop(columns=["product_id", "timestamp_ms"])
    X_test = X_test.drop(columns=["product_id", "timestamp_ms"])
    y_train = y_train.drop(columns=["product_id", "timestamp_ms"])
    y_test = y_test.drop(columns=["product_id", "timestamp_ms"])

    # Use TA-Lib to add technical indicators
    logger.debug("Calculating technical indicators")
    # experiment.log_parameter("technical_indicators", "Calculating technical indicators")
    X_train = add_technical_indicators(X_train)
    X_test = add_technical_indicators(X_test)

    # Use TA-Lib to add lagged features
    logger.debug("Calculating lagged features")
    # experiment.log_parameter("lagged_features", "Calculating lagged features")
    X_train = add_lagged_features(X_train)
    X_test = add_lagged_features(X_test)

    experiment.log_parameter('train_features', X_train.columns.tolist())
    experiment.log_parameter('test_features', X_test.columns.tolist())

    # Extract the indices from X-train where any of the technical indicators are NaN
    train_nan_indices = X_train.isna().any(axis=1)
    X_train = X_train.drop(X_train.index[train_nan_indices])
    y_train = y_train.drop(y_train.index[train_nan_indices])

    # Do the same for X_test
    test_nan_indices = X_test.isna().any(axis=1)
    X_test = X_test.drop(X_test.index[test_nan_indices])
    y_test = y_test.drop(y_test.index[test_nan_indices])

    # log the number of NaN rows and the percentage of dropped rows
    experiment.log_parameter("n_nan_rows_train", train_nan_indices.sum())
    logger.info(f"Number of NaN rows in train: {train_nan_indices.sum()}")

    experiment.log_parameter("n_nan_rows_test", test_nan_indices.sum())
    logger.info(f"Number of NaN rows in test: {test_nan_indices.sum()}")

    experiment.log_parameter("perc_dropped_rows_train", train_nan_indices.sum() / len(X_train) * 100)
    logger.info(f"Percentage of NaN rows in train: {train_nan_indices.sum() / len(X_train) * 100:.2f}%")

    experiment.log_parameter("perc_dropped_rows_test", test_nan_indices.sum() / len(X_test) * 100)
    logger.info(f"Percentage of NaN rows in test: {test_nan_indices.sum() / len(X_test) * 100:.2f}%")

    # Log the dimensions of the features and targets
    logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    experiment.log_parameter("X_train_shape", X_train.shape)
    experiment.log_parameter("y_train_shape", y_train.shape)
    experiment.log_parameter("X_test_shape", X_test.shape)
    experiment.log_parameter("y_test_shape", y_test.shape)

    experiment.log_metric("Average Target Price", y_test.mean())
    logger.info(f"Average Target Price: {y_test.mean()}")

    # Train a Baseline model
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    baseline_predictions = model.predict(X_test) # Assuming 'close' is the last known price
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    percentage_error = baseline_mae / y_test.mean() * 100

    # Log the baseline MAE and percentage error
    experiment.log_metric("Baseline MAE", baseline_mae)
    logger.info(f"Baseline MAE: {baseline_mae}")

    experiment.log_metric("Baseline Error %", percentage_error)   
    logger.info(f"Baseline % Error: {percentage_error:.2f}%")

    #Verify Basline predictions on the training data
    baseline_train_predictions = model.predict(X_train)
    baseline_train_mae = mean_absolute_error(y_train, baseline_train_predictions)
    baseline_train_percentage_error = baseline_train_mae / y_train.mean() * 100

    experiment.log_metric("Baseline Train MAE", baseline_train_mae)
    logger.info(f"Baseline Train MAE: {baseline_train_mae}")

    experiment.log_metric("Baseline Train Error %", baseline_train_percentage_error)   
    logger.info(f"Baseline Train % Error: {baseline_train_percentage_error:.2f}%")


    # Train a XGBoost model
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_percentage_error = xgb_mae / y_test.mean() * 100

    # Log the XGBoost MAE and percentage error
    experiment.log_metric("XGBoost MAE", xgb_mae)
    logger.info(f"XGBoost MAE: {xgb_mae}")

    experiment.log_metric("XGBoost Error %", xgb_percentage_error)   
    logger.info(f"XGBoost % Error: {xgb_percentage_error:.2f}%")

    # Verify XGB predictions on the training data
    xgb_train_predictions = xgb_model.predict(X_train)
    xgb_train_mae = mean_absolute_error(y_train, xgb_train_predictions)
    xgb_train_percentage_error = xgb_train_mae / y_train.mean() * 100

    # Log the XGBoost train MAE and percentage error
    experiment.log_metric("XGBoost Train MAE", xgb_train_mae)
    logger.info(f"XGBoost Train MAE: {xgb_train_mae}")

    experiment.log_metric("XGBoost Train Error %", xgb_train_percentage_error)   
    logger.info(f"XGBoost Train % Error: {xgb_train_percentage_error:.2f}%")



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
