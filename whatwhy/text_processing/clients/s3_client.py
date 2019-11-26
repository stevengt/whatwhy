import boto3
from .client import logger, BatchSourceBase, BatchDestinationBase

class S3ClientBase():

    def __init__(self, bucket_name, folder_name):
        logger.info(f"Connecting to S3 bucket/folder '{bucket_name}/{folder_name}'.")
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.folder_name = folder_name + "/"

class S3BatchSource(S3ClientBase, BatchSourceBase):
    """
    Retrieves batch files from AWS S3.

    AWS credentials should be stored in a format compatible with boto3,
    such as environment variables or a credentials file. For more information, see:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html

    Each instance of this class should only have ONE consumer.
    
    This class iterates through all the files available in the bucket/folder
    at the time of instantiation, and optionally deletes them as they are processed.
    """

    def __init__(self, bucket_name, folder_name, delete_when_complete=False):
        super().__init__(bucket_name, folder_name)
        self.delete_when_complete = delete_when_complete
        bucket_objects = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.folder_name)["Contents"]
        batch_keys = [ bucket_object["Key"] for bucket_object in bucket_objects if bucket_object["Key"] != self.folder_name ]
        batch_keys.sort()
        self.batch_iterator = iter(batch_keys)
        self.cur_batch_key = None

    def get_next_batch(self):
        try:
            self.cur_batch_key = self.batch_iterator.__next__()
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=self.cur_batch_key)
            return obj["Body"].read().decode("utf-8")
        except StopIteration as e:
            self.cur_batch_key = None
            raise e
        except Exception as e:
            logger.error(f"Failed to receive batch from S3: {e}")
            return None

    def mark_batch_as_complete(self):
        if self.delete_when_complete and self.cur_batch_key is not None:
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=self.cur_batch_key)
            except Exception as e:
                logger.error(f"Failed to delete batch from S3: {e}")
                self.cur_batch_key = None

class S3BatchDestination(S3ClientBase, BatchDestinationBase):
    """
    Writes batch files to AWS S3.
    
    AWS credentials should be stored in a format compatible with boto3,
    such as environment variables or a credentials file. For more information, see:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    """

    def __init__(self, bucket_name, folder_name):
        super().__init__(bucket_name, folder_name)

    def publish_batch_results(self, results, target_file_name=None):
        try:
            if target_file_name is None:
                raise Exception("No filename specified.")
            target_file_name = self.folder_name + target_file_name
            self.s3.put_object(Bucket=self.bucket_name, Key=target_file_name, Body=results)
        except Exception as e:
            logger.error(f"Failed to send batch to S3: {e}")
            self.cur_batch_key = None
