import os
from .client import logger, BatchSourceBase, BatchDestinationBase

class FileSystemBatchSource(BatchSourceBase):

    def __init__(self, folder_name, delete_when_complete=False):
        logger.info(f"Reading batches from local folder {folder_name}.")
        self.delete_when_complete = delete_when_complete
        self.folder_name = folder_name
        batch_file_names = os.listdir(folder_name)
        batch_file_names.sort()
        self.batch_file_names = iter(batch_file_names)
        self.cur_batch_file_name = None

    def get_next_batch(self):
        try:
            self.cur_batch_file_name = os.path.join(self.folder_name, self.batch_file_names.__next__())
            with open(self.cur_batch_file_name) as batch_file:
                return batch_file.read()
        except StopIteration as e:
            self.cur_batch_file_name = None
            raise e
        except Exception as e:
            logger.error(f"Failed to receive batch from local folder: {e}")

    def mark_batch_as_complete(self):
        if self.delete_when_complete and self.cur_batch_file_name is not None:
            try:
                os.remove(self.cur_batch_file_name)
            except Exception as e:
                logger.error(f"Failed to delete batch from local folder: {e}")

class FileSystemBatchDestination(BatchDestinationBase):

    def __init__(self, folder_name):
        logger.info(f"Writing batches to local folder {folder_name}.")
        self.folder_name = folder_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    def publish_batch_results(self, results, target_file_name):
        target_file_name = os.path.join(self.folder_name, target_file_name)
        try:
            with open(target_file_name, "w") as target_file:
                target_file.write(results)
        except Exception as e:
            logger.error(f"Failed to send batch to local folder: {e}")
