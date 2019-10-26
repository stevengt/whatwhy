import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

class BatchSourceBase():

    def get_next_batch(self):
        raise NotImplementedError()

    def mark_batch_as_complete(self):
        raise NotImplementedError()

class BatchDestinationBase():

    def publish_batch_results(self, results, target_file_name=None):
        raise NotImplementedError()

class BatchProcessorBase():
    """
    Retrieves batches of text to process from a BatchSource,
    and writes the results to a BatchDestination.
    """

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    def get_batch_results(self, batch):
        raise NotImplementedError()

    def run(self):
        while True:
            try:
                batch = self.source.get_next_batch()
                batch_results = self.get_batch_results(batch)
                results_file_content = batch_results["file_content"]
                target_file_name = batch_results["target_results_file_name"]
                self.dest.publish_batch_results(results_file_content, target_file_name)
                self.source.mark_batch_as_complete()
            except Exception as e:
                logger.error(e)
