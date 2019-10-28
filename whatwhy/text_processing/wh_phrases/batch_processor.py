import subprocess
from io import StringIO
import pandas as pd
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing import BatchProcessorBase, get_df_from_csv_string, get_csv_string_from_df

class WHPhrasesBatchProcessor(BatchProcessorBase):

    def __init__(self, source, dest):
        super().__init__(source, dest)
        self.corenlp_process = subprocess.Popen(["giveme5w1h-corenlp"])
        self.extractor = MasterExtractor()

    def get_top_wh_phrase(self, question_type, text_segment):
        doc = Document.from_text(text_segment)
        doc = self.extractor.parse(doc)
        try:
            return doc.get_top_answer(question_type).get_parts_as_text()
        except:
            return ""

    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        for question_type in QUESTION_WORDS:
            batch_as_df[question_type] = batch_as_df["Text"].map( lambda text_segment : self.get_top_wh_phrase(question_type, text_segment) )
        
        results_csv_string = get_csv_string_from_df(batch_as_df)

        results = {
            "target_results_file_name" : f"{batch_as_df.index[0]}.csv",
            "file_content" : results_csv_string
        }
        return results
