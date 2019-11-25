import logging
import pandas as pd
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string, get_csv_string_from_df

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

class BatchConsolidator (BatchProcessorBase):
    """Consolidates data from multiple CSV files into a single CSV file."""

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name=None,
                       dest_col_name=None,
                       include_cols=None):
        super().__init__(source, dest, id_col_name=id_col_name)

    def run(self):
        df = pd.DataFrame()
        while True:
            try:
                batch = self.source.get_next_batch()
                batch_df = get_df_from_csv_string(batch)
                df = pd.concat([df, batch_df])
                self.source.mark_batch_as_complete()
            except StopIteration:
                logger.info(f"Finished reading batches from source.")
                results_file_content = get_csv_string_from_df(df)
                target_file_name = "consolidated_batches.csv"
                self.dest.publish_batch_results(results_file_content, target_file_name)
                break
            except Exception as e:
                logger.error(e)
                break
