import subprocess
from time import sleep
import numpy as np
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string, get_csv_string_from_df

class WHPhrasesBatchProcessor(BatchProcessorBase):

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Preprocessed Text",
                       include_cols=["Tweet ID"]):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            include_cols=include_cols)

        sleep(60)
        extractor_preprocessor = Preprocessor("http://corenlp-service:9000")
        self.extractor = MasterExtractor(preprocessor=extractor_preprocessor)

    def get_top_wh_phrase(self, question_type, text_segment):
        if text_segment is None or text_segment is np.nan:
            return None
        try:
            doc = Document.from_text(text_segment)
            doc = self.extractor.parse(doc)
            return doc.get_top_answer(question_type).get_parts_as_text()
        except:
            return None

    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        for question_type in QUESTION_WORDS:
            batch_as_df[question_type] = batch_as_df[self.source_col_name].map( lambda text_segment : self.get_top_wh_phrase(question_type, text_segment) )
        
        results_df_cols = [self.id_col_name]
        results_df_cols.extend(QUESTION_WORDS)
        results_df_cols.extend(self.include_cols)
        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results
