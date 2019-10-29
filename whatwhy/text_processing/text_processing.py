import logging
from io import StringIO
import csv
import pandas as pd

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def get_csv_string_from_df(df):
    with StringIO() as csv_stream:
        df.to_csv(csv_stream, sep="\t", quoting=csv.QUOTE_ALL, quotechar='"')
        return csv_stream.getvalue()

def get_df_from_csv_string(csv_string):
    with StringIO(csv_string) as csv_stream:
        return pd.read_csv(csv_stream, index_col=False, sep="\t", dtype=str, quoting=csv.QUOTE_ALL, quotechar='"')

def get_df_from_file(file_name):
    return pd.read_csv(file_name, index_col=False, sep="\t", dtype=str, quoting=csv.QUOTE_ALL, quotechar='"')

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
