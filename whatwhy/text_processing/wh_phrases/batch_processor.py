from io import StringIO
import pandas as pd
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing import BatchProcessorBase, SQSClient, S3Client

class WHPhrasesBatchProcessor(BatchProcessorBase):

    def __init__(self):
        self.extractor = MasterExtractor()
        self.source = SQSClient(queue_name="whatwhy-processing")
        self.dest = S3Client(bucket_name="whatwhy-data", folder_name="batch-results")

    def get_top_wh_phrase(self, question_type, text_segment):
        doc = Document.from_text(text_segment)
        doc = self.extractor.parse(doc)
        try:
            return doc.get_top_answer(question_type).get_parts_as_text()
        except:
            return ""

    def get_batch_results(self, batch):
        batch_as_csv = StringIO(batch)
        batch_as_df = pd.read_csv(batch_csv)
        for question_type in QUESTION_WORDS:
            batch_as_df[question_type] = batch_as_df["Text"].map( lambda text_segment : self.get_top_wh_phrase(question_type, text_segment) )
        
        results_csv_string = StringIO()
        batch_as_df.to_csv(results_csv_string)

        results = {
            "target_results_file_name" : f"{batch_as_df.index.iloc[0]}.csv",
            "file_content" : results_csv_string.get_value()
        }
        return results

def main():
    corenlp_process = Popen(["giveme5w1h-corenlp"])
    batch_processor = WHPhrasesBatchProcessor()
    batch_processor.run()

if __name__ == "__main__":
    main()
