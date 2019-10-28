import logging
import pandas as pd

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

    def populate_from_df(self, df, batch_size=1000):
        nrows = df.shape[0]
        batches = [ df.iloc[i:i+batch_size] for i in range(0, nrows, batch_size) ]
        for i, batch in enumerate(batches):
            try:
                target_file_name = f"batch{i}.csv"
                csv_string = StringIO()
                batch.to_csv(csv_string)
                self.publish_batch_results(csv_string, target_file_name=target_file_name)
            except Exception as e:
                logger.error(f"Failed to populate batch {i}: {e}")

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
