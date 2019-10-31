from time import sleep
from multiprocessing import Process, Manager
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

    def get_top_wh_phrases(self, text_segment, top_phrases):
        for question_type in QUESTION_WORDS:
            top_phrases[question_type] = None

        if text_segment is not None and text_segment is not np.nan:
            try:
                doc = Document.from_text(text_segment)
                doc = self.extractor.parse(doc)
                for question_type in QUESTION_WORDS:
                    try:
                        top_phrases[question_type] = doc.get_top_answer(question_type).get_parts_as_text()
                    except:
                        continue
            except:
                pass

    def get_batch_results(self, batch):
        with Manager() as manager:
            batch_as_df = get_df_from_csv_string(batch)
            for question_type in QUESTION_WORDS:
                batch_as_df[question_type] = None
            for i, row in batch_as_df.iterrows():
                top_wh_phrases = manager.dict()
                top_wh_phrases_process = Process(target=self.get_top_wh_phrases, args=(row[self.source_col_name], top_wh_phrases))
                top_wh_phrases_process.start()
                top_wh_phrases_process.join(timeout=15)
                top_wh_phrases_process.terminate()
                if top_wh_phrases_process.exitcode is None:
                    raise TimeoutError("Failed to process text-segment in 15 seconds.")

                for question_type in QUESTION_WORDS:
                    batch_as_df.at[i, question_type] = top_wh_phrases.get(question_type)

            results_df_cols = [self.id_col_name]
            results_df_cols.extend(QUESTION_WORDS)
            results_df_cols.extend(self.include_cols)
            results_df = batch_as_df[results_df_cols]
            results_csv_string = get_csv_string_from_df(results_df)

            results = {
                "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
                "file_content" : results_csv_string
            }
            return results
