from typing import List

import hopsworks
import pandas as pd

from src.config import hopsworks_config as config

# Connect to Hopsworks project
project = hopsworks.login(
    project=config.hopsworks_project_name,
    api_key_value=config.hopsworks_api_key,
)

# Get the feature store
feature_store = project.get_feature_store()


def write_to_feature_group(
    value: str,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
):
    """
    Writes the given `value` to the `feature_group_name` in the feature store

    Args:
        value (dict): The data to write to the feature store
        feature_group_name (str): The name of the feature group in the feature store
        feature_group_version (int): The version of the feature group
        feature_group_primary_key (List[str]): The primary key of the feature group
        feature_group_event_time (str): The event time column of the feature group
        start_offline_materialization (bool): Whether to start offline materialization or not when we save the `value` to the feature group

    Returns:
        None
    """
    try:
        feature_group = feature_store.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            primary_key=feature_group_primary_keys,
            event_time=feature_group_event_time,
            online_enabled=True,
            # Great Expecations to validate the data before saving features to the feature store
            # expectation_suite=expectation_suite,
        )
    except Exception as e:
        print(f'Error creating or getting feature group: {e}')
        return

    # Convert the value to a DataFrame
    value_df = pd.DataFrame([value])

    # breakpoint()

    # Write the data to the feature group
    feature_group.insert(
        value_df,
        write_options={'start_offline_materialization': start_offline_materialization},
    )
