from time import sleep
import numpy as np
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.extractors import action_extractor, cause_extractor, method_extractor
from whatwhy import QUESTION_WORDS
from whatwhy.resource_manager.nltk import configure_nltk
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string, get_csv_string_from_df

class WHPhrasesBatchProcessor(BatchProcessorBase):
    """
    Extracts the WH phrases (who, what, when, where, why, how) from text.

    This is intended to be run from within a Docker network, since
    access to a Stanford CoreNLP server API at http://corenlp-service:9000
    is required. Please see the readme file at https://github.com/stevengt/whatwhy
    for more information.
    """

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Preprocessed Text",
                       dest_col_name=None,
                       include_cols=None):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            include_cols=include_cols)
        configure_nltk()
        sleep(60) # Wait for Stanford CoreNLP server to start.
        extractor_preprocessor = Preprocessor("http://corenlp-service:9000")
        extractors = [
            action_extractor.ActionExtractor(),
            cause_extractor.CauseExtractor(),
            method_extractor.MethodExtractor()
        ]
        self.extractor = MasterExtractor(preprocessor=extractor_preprocessor, extractors=extractors)

    def get_top_wh_phrases(self, text_segment):
        top_phrases = {}
        for question_type in QUESTION_WORDS:
            top_phrases[question_type] = None

        if text_segment is not None and text_segment is not np.nan:
            try:
                doc = Document.from_text(text_segment)
                doc = self.extractor.parse(doc)
                for question_type in QUESTION_WORDS:
                    if question_type == "where" or question_type == "when":
                        top_phrases[question_type] = "NOT PROCESSED"
                    else:
                        try:
                            top_phrases[question_type] = doc.get_top_answer(question_type).get_parts_as_text()
                        except:
                            continue
            except:
                pass

        return top_phrases

    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        for question_type in QUESTION_WORDS:
            batch_as_df[question_type] = None
        for i, row in batch_as_df.iterrows():
            top_wh_phrases = self.get_top_wh_phrases(row[self.source_col_name])
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
