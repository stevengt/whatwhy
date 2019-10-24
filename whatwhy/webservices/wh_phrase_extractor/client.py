"""
Client for extracting the WH-phrases ('who', 'what', 'when', 'where', 'why', 'how')
from raw text.

This client requires the URL of a deployed instance of the
WHPhraseExtractorServer, which is assumed to be at http://localhost:9099 by default.

The client/server in this package serve as a wrapper around some of the
functionality of Giveme5W1H: https://github.com/fhamborg/Giveme5W1H

Additionally, this package enables "batch" processing by concatenating
multiple text-segments and processing them as a single text-segment.
"""

import json
import requests
from .logger import logger
from .text_processing import TextSegment, TextSegmentBatchResults
from whatwhy import QUESTION_WORDS

class WHPhraseExtractorClient():


    def __init__( self,
                  server_url="http://localhost:9099",
                  batch_size=1,
                  max_num_candidate_phrases=1,
                  request_timeout_in_seconds=5 ):

        self.api_endpoint_url = f"{server_url}/get-wh-phrases"
        self.batch_size = batch_size
        self.max_num_candidate_phrases = max_num_candidate_phrases
        self.request_timeout_in_seconds = request_timeout_in_seconds

    def get_request_body_json(self, text_segments):
        json_fields = {
            "batch-size" : self.batch_size,
            "max-num-candidate-phrases" : self.max_num_candidate_phrases,
            "text-segments" : [ text_segment.as_dict() for text_segment in text_segments ]
        }
        return json.dumps(json_fields).encode("utf-8")

    def get_wh_phrases_from_text_segments(self, text_segments):
        """
        Input: A list of text_processing.TextSegment objects.
        Output: A text_processing.TextSegmentBatchResults object.
        """
        results = []
        try:
            request_body = self.get_request_body_json(text_segments)
            response = requests.post( self.api_endpoint_url, data=request_body, timeout=self.request_timeout_in_seconds )
            if response.ok:
                results_as_dict = json.loads( response.content, encoding="utf-8", strict=True )
                results = [ TextSegmentBatchResults(result["id's"], result["wh-phrases"]) for result in results_as_dict ]
            else:
                raise Exception(f"HTTP {response.status_code}: {response.content}")
        except Exception as e:
            logger.error(f"Unable to get WH-phrases from WHPhraseExtractorServer: {e}")
        return results

    def get_wh_phrases_from_raw_texts(self, raw_text_list):
        """
        Input: A list of strings.
        Output: A text_processing.TextSegmentBatchResults object.
        """
        text_segments = [ TextSegment(content=raw_text) for raw_text in raw_text_list ]
        return self.get_wh_phrases_from_text_segments(text_segments)

    def get_wh_phrases_from_df(self, df, content_col_name, id_col_name=None):
        """
        Input: A pandas DataFrame and the column names of:
            - The text content to process
            - (Optional) ID's
        Output: A text_processing.TextSegmentBatchResults object.
        """
        text_segments = df.apply( lambda row: TextSegment( row[content_col_name], row[id_col_name] ), axis=1 ) \
                          .tolist()
        return self.get_wh_phrases_from_text_segments(text_segments)
