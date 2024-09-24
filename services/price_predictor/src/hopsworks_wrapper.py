from typing import Optional
import time

import pandas as pd
import hopsworks
from hsfs.feature_view import FeatureView
from hsfs.feature_store import FeatureStore

from src.config import HopsworksConfig


class HopsworksWrapper:
    """
    A class for reading OHLC data from the feature store
    """

    def __init__(
        self,
        ohlc_window_sec: int,
        hopsworks_config: HopsworksConfig,
        feature_view_name: str,
        feature_view_version: int,
        feature_group_name: Optional[str] = None,
        feature_group_version: Optional[int] = None,
    ):
        self.ohlc_window_sec = ohlc_window_sec
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version

        self._feature_store = self._get_feature_store(hopsworks_config)

    def _get_feature_view(self) -> FeatureView:
        """
        Returns the feature view object that reads data from the feature store
        """
        if self.feature_group_name is None:
            try:
                return self._feature_store.get_feature_view(
                    name=self.feature_view_name,
                    version=self.feature_view_version,
                )
            except Exception:
                raise ValueError(
                    "The feature group name and version must be provided if the feature view does not exist."
                )

        # Get the feature group
        feature_group = self._feature_store.get_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
        )

        # Get or create the feature view
        feature_view = self._feature_store.get_or_create_feature_view(
            name=self.feature_view_name,
            version=self.feature_view_version,
            query=feature_group.select_all(),
        )

        possibly_different_feature_group = (
            feature_view.get_parent_feature_groups().accessible[0]
        )

        if (
            possibly_different_feature_group.name != feature_group.name
            or possibly_different_feature_group.version != feature_group.version
        ):
            raise ValueError(
                "The feature view and feature group names and versions do not match."
            )

        return feature_view

    def read_from_offline_store(
        self,
        product_id: str,
        last_n_days: int,
    ) -> pd.DataFrame:
        """
        Reads OHLC data from the offline feature store for the given product_id
        """
        to_timestamp_ms = int(time.time() * 1000)
        from_timestamp_ms = to_timestamp_ms - last_n_days * 24 * 60 * 60 * 1000

        feature_view = self._get_feature_view()
        features = feature_view.get_batch_data()

        # filter the features for the given product_id and time range
        features = features[features["product_id"] == product_id]
        features = features[features["timestamp_ms"] >= from_timestamp_ms]
        features = features[features["timestamp_ms"] <= to_timestamp_ms]
        # sort the features by timestamp (ascending)
        features = features.sort_values(by="timestamp_ms").reset_index(drop=True)

        # breakpoint()

        return features

    @staticmethod
    def _get_feature_store(hopsworks_config: HopsworksConfig) -> FeatureStore:
        """
        Returns the feature store object that we will use to read our OHLC data.
        """
        project = hopsworks.login(
            project=hopsworks_config.hopsworks_project_name,
            api_key_value=hopsworks_config.hopsworks_api_key,
        )

        return project.get_feature_store()
