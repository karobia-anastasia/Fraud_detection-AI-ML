from kafka import KafkaProducer
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaTransactionProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Initialize Kafka producer.
        """
        try:
            # Initialize the Kafka producer only once
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Serialize the data as JSON
            )
            logger.info("Kafka Producer initialized.")
        except Exception as e:
            logger.error(f"Error initializing Kafka Producer: {e}")
            raise

    def send_transaction(self, transaction, topic='transactions'):
        """
        Send a transaction to the Kafka topic.
        """
        try:
            # Send the transaction to the specified Kafka topic
            self.producer.send(topic, transaction)
            self.producer.flush()  # Ensure all messages are sent before closing
            logger.info(f"Transaction sent to Kafka topic '{topic}': {transaction}")
        except Exception as e:
            logger.error(f"Error sending transaction to Kafka: {e}")

    def close(self):
        """
        Close the Kafka producer.
        """
        try:
            self.producer.close()  # Close the producer to free resources
            logger.info("Kafka Producer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka Producer: {e}")

# Example usage
if __name__ == "__main__":
    transaction = {
        'amount': 150.75,
        'user_id': 'user123',
        'timestamp': '2023-09-29T12:34:56',
        'transaction_frequency': 3.5
    }

    # Initialize Kafka producer
    producer = KafkaTransactionProducer()

    # Send transaction
    producer.send_transaction(transaction)

    # Close the producer
    producer.close()
