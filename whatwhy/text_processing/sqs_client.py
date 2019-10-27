import boto3
from .text_processing import BatchSourceBase, BatchDestinationBase

class SQSBatchSource(BatchSourceBase):
    """
    Retrieves messages from AWS SQS.
    Each instance of this class should only have ONE consumer.
    """
    
    def __init__(self, queue_name, region_name="us-east-2"):
        self.sqs = boto3.client("sqs", region_name=region_name)
        self.queue_name = queue_name
        self.queue_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
        self.message = None

    def get_next_batch(self):
        try:
            messages = self.sqs.receive_message(QueueUrl=self.queue_url, MaxNumberOfMessages=1)
            if "Messages" in messages:
                self.message = messages["Messages"][0]
                return self.message["Body"]
        except:
            return None

    def mark_batch_as_complete(self):
        try:
            message_receipt_handle = self.message["ReceiptHandle"]
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=message_receipt_handle)
        except:
            return

class SQSBatchDestination(BatchDestinationBase):

    def __init__(self, queue_name, region_name="us-east-2"):
        self.sqs = boto3.client("sqs", region_name=region_name)
        self.queue_name = queue_name
        self.queue_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]

    def publish_batch_results(self, results, target_file_name=None):
        try:
            self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=results)
        except:
            return
