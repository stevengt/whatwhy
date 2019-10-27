import boto
from boto.s3.key import Key
from .text_processing import BatchDestinationBase

class S3BatchSource(BatchSourceBase):
    
    def __init__(self, bucket_name, folder_name):
        self.s3 = boto3.resource("s3")
        self.bucket_name
        self.bucket = s3.Bucket(bucket_name)
        self.folder_name = folder_name
        self.batch_iterator = self.bucket.list(folder_name)

    def get_next_batch(self):
        try:
            key = self.batch_iterator.__next__()
            obj = self.s3.Object(self.bucket_name, key)
            return obj.get()['Body'].read().decode('utf-8')
        except:
            return None

    def mark_batch_as_complete(self):
        return

class S3BatchDestination(BatchDestinationBase):

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
