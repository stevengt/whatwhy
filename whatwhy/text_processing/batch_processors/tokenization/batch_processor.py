import numpy as np
from textblob import TextBlob
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string, get_csv_string_from_df

class BatchTokenizer(BatchProcessorBase):

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Preprocessed Text",
                       dest_col_name="Tokens",
                       include_cols=None):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            dest_col_name=dest_col_name,
                            include_cols=include_cols)
        
    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        batch_as_df[self.dest_col_name] = batch_as_df[self.source_col_name].str.lower()
        batch_as_df[self.dest_col_name] = batch_as_df[self.dest_col_name].apply( self.get_list_of_lemmatized_words_from_text )

        results_df_cols = [self.id_col_name, self.dest_col_name]
        results_df_cols.extend(self.include_cols)
        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results

    def get_list_of_lemmatized_words_from_text(self, text):
        if text is None or text is np.nan:
            return None
        try:
            text_blob = TextBlob(text)
            return list( text_blob.words.singularize().lemmatize() )
        except:
            return None
