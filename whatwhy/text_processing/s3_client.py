import boto
from boto.s3.key import Key
from .text_processing import BatchDestinationBase

class S3Client(BatchDestinationBase):

    def __init__(self, bucket_name, folder_name):
        s3 = boto3.resource("s3")
        self.bucket = s3.Bucket(bucket_name)
        self.folder_name = folder_name

    def publish_batch_results(self, results, target_file_name=None):
        if target_file_name is None:
            target_file_name = "unspecified-filename.txt"
        k = Key(self.bucket)
        k.key = f"{self.folder_name}/{target_file_name}"
        k.set_contents_from_string(results)
