from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string

class BatchTransferer(BatchProcessorBase):

    def __init__(self, source, dest, id_col_name="ID"):
        super().__init__(source, dest, id_col_name=id_col_name)
    
    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : batch
        }
        return results
