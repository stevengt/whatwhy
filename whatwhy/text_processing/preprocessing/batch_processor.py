import re
import numpy as np
from gingerit.gingerit import GingerIt
from textblob import TextBlob
from whatwhy.text_processing import BatchProcessorBase

class BatchPreprocessor(BatchProcessorBase):

    def __init__(self, source, dest):
        super().__init__(source, dest)
        self.spelling_and_grammar_parser = GingerIt()

    def get_batch_results(self, batch):
        batch_as_csv = StringIO(batch)
        batch_as_df = pd.read_csv(batch_csv)
        batch_as_df["Preprocessed Text"] = batch_as_df["Text"].apply(self.remove_reply_tag_from_tweet_text) \
                                                              .apply(self.autocorrect_spelling_and_grammar)
        results_df = batch_as_df[["Tweet ID", "Preprocessed Text"]]
        results_csv_string = StringIO()
        results_df.to_csv(results_csv_string, index=False)
        return results_csv_string.getvalue()

    def remove_reply_tag_from_tweet_text(text):
        if text is None or text is np.nan:
            return text
        reply_tag = re.match("@[^\s]+[\s]+", text)
        if reply_tag is not None:
            reply_tag = reply_tag.group(0)
            text = text.replace(reply_tag, "")
        return text

    def autocorrect_spelling_and_grammar(text):
        try:
            return self.spelling_and_grammar_parser.parse(text)["result"]
        except:
            return text
