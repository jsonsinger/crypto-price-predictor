from datetime import timedelta

from loguru import logger
from quixstreams import Application

from src.config import ohlc_config as config


def transform_trade_to_ohlcv(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    ohlc_window: int,
):
    """
    Reads trades from the given `kafka_input_topic`, transforms them into OHLC data, and publishes them to the `kafka_output_topic`

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_input_topic (str): The Kafka topic to read the trades from
        kafka_output_topic (str): The Kafka topic to publish the OHLC data to
        kafka_consumer_group (str): The Kafka consumer group
        ohlc_window (int): The window size in seconds for the OHLC data

    Returns:
        None
    """

    # Create an Application instance with Kakfa
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json')
    output_topic = app.topic(name=kafka_output_topic, value_serializer='json')

    # Create a Quicstreams DataFrame
    sdf = app.dataframe(input_topic)

    sdf = (
        sdf.tumbling_window(duration_ms=timedelta(seconds=ohlc_window))
        .reduce(reducer=ohlc_reducer, initializer=ohlc_initializer)
        .final()
        # .current()
    )

    # Unpack the Quicstreams DataFrame
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    sdf['product_id'] = sdf['value']['product_id']
    sdf['timestamp_ms'] = sdf['end']

    # Keep only the necessary columns
    sdf = sdf[['open', 'high', 'low', 'close', 'volume', 'timestamp_ms', 'product_id']]

    sdf.update(logger.debug)

    # Publish the OHLC data to the output topic
    sdf.to_topic(output_topic)

    app.run(sdf)


def ohlc_initializer(trade: dict):
    """
    Initialize the OHLC data with the first trade
    """
    return {
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['quantity'],
        'product_id': trade['product_id'],
    }


def ohlc_reducer(candle: dict, trade: dict):
    """
    Update the OHLC data with the latest trade
    """
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity']
    candle['product_id'] = trade['product_id']

    return candle


if __name__ == '__main__':
    transform_trade_to_ohlcv(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        ohlc_window=config.ohlc_window,
    )
