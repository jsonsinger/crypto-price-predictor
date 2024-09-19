from typing import List

from loguru import logger
from quixstreams import Application

from src.config import config
from src.kraken_websocket_api import KrakenWebsocketAPI, Trade


def produce_trades(
    kafka_broker_address: str,
    kafka_topic: str,
    product_ids: List[str],
):
    """
    Reads trades from the Kraken Websocket API and publishes them to the given `kafka_topic`

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_topic (str): The Kafka topic to publish the trades to
        product_ids (List[str]): The product IDs to get trades for

    Returns:
        None
    """

    # Create an Application instance with Kakfa
    app = Application(broker_address=kafka_broker_address)

    # Define a topic `kafka_topic` with JSON serialization
    topic = app.topic(name=kafka_topic, value_serializer='json')

    # Create a KrakenWebsocketAPI instance
    kraken_api = KrakenWebsocketAPI(product_ids=product_ids)

    # Create a Producer instance
    with app.get_producer() as producer:
        while not kraken_api.is_done():
            # Get the latest batch of trades
            trades: List[Trade] = kraken_api.get_trades()

            for trade in trades:
                # Serialize the trade and publish it to the topic
                # transform it into a sequence of bytes
                message = topic.serialize(
                    key=trade.product_id, value=trade.model_dump()
                )

                # Produce a message into the Kafka topic
                producer.produce(topic=topic.name, value=message.value, key=message.key)

                logger.debug(f'Pushed trade to Kafka: {trade}')


if __name__ == '__main__':
    try:
        produce_trades(
            kafka_broker_address=config.kafka_broker_address,
            kafka_topic=config.kafka_topic,
            product_ids=config.product_ids,
        )
    except KeyboardInterrupt:
        logger.info(
            'Trade Producer is shutting down! Time to let the service cool off.'
        )
