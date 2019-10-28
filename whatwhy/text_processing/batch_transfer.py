from .text_processing import BatchProcessorBase

class BatchTransferer(BatchProcessorBase):

    def __init__(self, source, dest):
        super().__init__(source, dest)
    
    def get_batch_results(self, batch):
        return batch
