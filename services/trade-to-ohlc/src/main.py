from datetime import timedelta

from loguru import logger
from quixstreams import Application


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
    sdf['timestamp_ms'] = sdf['end']

    # Keep only the necessary columns
    sdf = sdf[['open', 'high', 'low', 'close', 'volume', 'timestamp_ms']]

    sdf.update(logger.debug)

    # Publish the OHLC data to the output topic
    sdf.to_topic(output_topic)

    app.run(sdf)


def ohlc_initializer(trade: dict):
    logger.info(trade)
    """
    Initialize the OHLC data with the first trade
    """
    return {
        #'product_id': trade['product_id'],
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['quantity'],
        #'timestamp_ms': trade['timestamp_ms'],
    }


def ohlc_reducer(candle: dict, trade: dict):
    """
    Update the OHLC data with the latest trade
    """
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity']
    # candle['timestamp_ms'] = trade['timestamp_ms']

    return candle


if __name__ == '__main__':
    transform_trade_to_ohlcv(
        kafka_broker_address='localhost:19092',
        kafka_input_topic='trades',
        kafka_output_topic='ohlcv',
        kafka_consumer_group='trade-to-ohlcv',
        ohlc_window=60,
    )
