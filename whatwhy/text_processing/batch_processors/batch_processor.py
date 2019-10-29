import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

class BatchProcessorBase():
    """
    Retrieves batches of text to process from a BatchSource,
    and writes the results to a BatchDestination.
    """

    def __init__(self, source,
                       dest,
                       id_col_name=None,
                       source_col_name=None,
                       dest_col_name=None,
                       include_cols=None):
        self.source = source
        self.dest = dest
        self.id_col_name = id_col_name
        self.source_col_name = source_col_name
        self.dest_col_name = dest_col_name
        self.include_cols = include_cols if include_cols is not None else []

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
            except StopIteration:
                logger.info(f"Finished reading batches from source.")
                break
            except Exception as e:
                logger.error(e)
