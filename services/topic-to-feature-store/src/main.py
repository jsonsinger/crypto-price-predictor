import json
from typing import List

from loguru import logger
from quixstreams import Application

from src.config import fs_config as config
from src.hopswork_api import write_to_feature_group


def topic_to_feature_store(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
    batch_size: int,
):
    """
    Reads data from the given `kafka_input_topic` and writes it to `feature_group_name` in the feature store

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_input_topic (str): The Kafka topic to read the data from
        kafka_consumer_group (str): The Kafka consumer group
        feature_group_name (str): The name of the feature group in the feature store
        feature_group_version (int): The version of the feature group
        feature_group_primary_keys (List[str]): The primary keys of the feature group
        feature_group_event_time (str): The event time column of the feature group
        start_offline_materialization (bool): Whether to start offline materialization or not when we save the data to the feature group
        batch_size (int): The number of rows to write to the feature group at once

    Returns:
        None
    """

    # Create an Application instance with Kakfa
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    batch = []

    with app.get_consumer() as consumer:
        consumer.subscribe(topics=[kafka_input_topic])

        while True:
            msg = consumer.poll(timeout=0.1)

            if msg is None:
                continue
            elif msg.error():
                logger.error(f'Consumer error: {msg.error()}')
                continue

            # Decode the message value
            value = msg.value().decode('utf-8')

            # Parse the value into a dictionary
            value = json.loads(value)

            # Append the value to the batch
            batch.append(value)

            # If the batch is not full keep polling
            if len(batch) < batch_size:
                # TODO - cater for partial batches over an elapsed period of time
                logger.debug(
                    f'Batch size: {len(batch)} of {batch_size}. Continue polling...'
                )
                continue

            logger.debug(
                f'Batch size: {len(batch)} of {batch_size}. Pushing to Feature Store.'
            )
            # Write the data to the feature group
            write_to_feature_group(
                batch,
                feature_group_name,
                feature_group_version,
                feature_group_primary_keys,
                feature_group_event_time,
                start_offline_materialization,
            )

            # Clear the batch
            batch = []


if __name__ == '__main__':
    try:
        topic_to_feature_store(
            kafka_broker_address=config.kafka_broker_address,
            kafka_input_topic=config.kafka_input_topic,
            kafka_consumer_group=config.kafka_consumer_group,
            feature_group_name=config.feature_group_name,
            feature_group_version=config.feature_group_version,
            feature_group_primary_keys=config.feature_group_primary_keys,
            feature_group_event_time=config.feature_group_event_time,
            start_offline_materialization=config.start_offline_materialization,
            batch_size=config.batch_size,
        )
    except KeyboardInterrupt:
        logger.info(
            'Topic to Feature Store is shutting down! Time to let the service cool off.'
        )
