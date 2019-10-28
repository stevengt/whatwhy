from .text_processing import BatchProcessorBase, get_df_from_csv_string

class BatchTransferer(BatchProcessorBase):

    def __init__(self, source, dest):
        super().__init__(source, dest)
    
    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        results = {
            "target_results_file_name" : f"batch{batch_as_df.index.iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return batch
