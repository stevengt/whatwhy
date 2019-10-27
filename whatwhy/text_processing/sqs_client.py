import boto3
from .text_processing import BatchSourceBase

class SQSBatchSource(BatchSourceBase):
    """
    Retrieves messages from AWS SQS.
    Each instance of this class should only have ONE consumer.
    """
    
    def __init__(self, queue_name):
        sqs = boto3.client('sqs')
        self.queue = sqs.get_queue_by_name(QueueName=queue_name)

    def get_next_batch(self):
        messages = self.queue.receive_message(MaxNumberOfMessages=1)
        if "Messages" in messages:
            self.message = messages[0]
            return self.message["Body"]

    def mark_batch_as_complete(self):
        message_receipt_handle = self.message["ReceiptHandle"]
        self.queue.delete_message(ReceiptHandle=message_receipt_handle)

class SQSBatchDestination(BatchDestinationBase):

    def __init__(self, queue_name):
        sqs = boto3.client('sqs')
        self.queue = sqs.get_queue_by_name(QueueName=queue_name)

    def publish_batch_results(self, results, target_file_name=None):
        self.queue.send_message(MessageBody=results)
