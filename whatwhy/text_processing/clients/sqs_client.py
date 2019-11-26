import boto3
from .client import logger, BatchSourceBase, BatchDestinationBase

class SQSClientBase():

    def __init__(self, queue_name, region_name="us-east-1"):
        logger.info(f"Connecting to SQS queue with name '{queue_name}' in region '{region_name}'.")
        self.sqs = boto3.client("sqs", region_name=region_name)
        self.queue_name = queue_name
        self.queue_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]

class SQSBatchSource(SQSClientBase, BatchSourceBase):
    """
    Retrieves batch files from AWS SQS.
    
    AWS credentials should be stored in a format compatible with boto3,
    such as environment variables or a credentials file. For more information, see:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    
    Each instance of this class should only have ONE consumer.
    """
    
    def __init__(self, queue_name, region_name="us-east-1"):
        super().__init__(queue_name, region_name)
        self.message = None

    def get_next_batch(self):
        try:
            messages = self.sqs.receive_message(QueueUrl=self.queue_url, MaxNumberOfMessages=1)
            if "Messages" in messages:
                self.message = messages["Messages"][0]
                return self.message["Body"]
        except Exception as e:
            logger.error(f"Failed to receive batch from SQS: {e}")
            return None

    def mark_batch_as_complete(self):
        try:
            message_receipt_handle = self.message["ReceiptHandle"]
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=message_receipt_handle)
        except Exception as e:
            logger.error(f"Failed to delete batch from SQS: {e}")
            return

class SQSBatchDestination(SQSClientBase, BatchDestinationBase):
    """
    Writes batch files to AWS SQS.

    AWS credentials should be stored in a format compatible with boto3,
    such as environment variables or a credentials file. For more information, see:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    """

    def __init__(self, queue_name, region_name="us-east-1"):
        super().__init__(queue_name, region_name)

    def publish_batch_results(self, results, target_file_name=None):
        try:
            self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=results)
        except Exception as e:
            logger.error(f"Failed to send batch to SQS: {e}")
            return
