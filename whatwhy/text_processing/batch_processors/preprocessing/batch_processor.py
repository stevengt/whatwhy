import re
import numpy as np
import jamspell
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_csv_string_from_df, get_df_from_csv_string
from whatwhy.resource_manager import get_jamspell_model_file_name

def get_spell_checker():    
    spell_checker = jamspell.TSpellCorrector()
    spell_checker.LoadLangModel(get_jamspell_model_file_name())
    return spell_checker

class BatchPreprocessor(BatchProcessorBase):

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Text",
                       dest_col_name="Preprocessed Text",
                       include_cols=None):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            dest_col_name=dest_col_name,
                            include_cols=include_cols)
        self.spell_checker = get_spell_checker()

    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        batch_as_df[self.dest_col_name] = batch_as_df[self.source_col_name].apply( self.remove_reply_tag_from_tweet_text) \
                                                                           .apply( self.remove_url ) \
                                                                           .apply( self.autocorrect_spelling )
        results_df_cols = [self.id_col_name, self.dest_col_name]
        results_df_cols.extend(self.include_cols)
        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results

    def remove_url(self, text):
        if text is None or text is np.nan:
            return text
        try:
            url_regex = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
            text = re.sub(url_regex, "", text, flags=re.MULTILINE) 
            return text 
        except:
            return text

    def remove_reply_tag_from_tweet_text(self, text):
        if text is None or text is np.nan:
            return text
        try:
            reply_tag = re.match("@[^\s]+[\s]+", text)
            if reply_tag is not None:
                reply_tag = reply_tag.group(0)
                text = text.replace(reply_tag, "")
            return text
        except:
            return text

    def autocorrect_spelling(self, text):
        if text is None or text is np.nan:
            return text    
        try:
            return self.spell_checker.FixFragment(text)
        except:
            return text
