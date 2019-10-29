import logging
from whatwhy.text_processing.helper_methods import get_csv_string_from_df

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
                csv_string = get_csv_string_from_df(batch)
                self.publish_batch_results(csv_string, target_file_name=target_file_name)
            except Exception as e:
                logger.error(f"Failed to populate batch {i}: {e}")
