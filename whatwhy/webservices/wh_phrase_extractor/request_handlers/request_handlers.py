
from flask import current_app, jsonify
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from Giveme5W1H.extractor.extractor import MasterExtractor
from .request_params import RequestParams
from whatwhy.webservices.wh_phrase_extractor.text_processing import TextSegmentBatchProcessor

def get_giveme5w1h_extractor():
    extractor_preprocessor = Preprocessor(current_app.config["corenlp_server_url"])
    return MasterExtractor(preprocessor=extractor_preprocessor)

class RequestHandler():
    def __init__(self, request_type):
        self.request_params = RequestParams.get_current_request_params(request_type)

class GetWHPhrasesRequestHandler(RequestHandler):

    def __init__(self):
        super().__init__(request_type="get-wh-phrases")
        self.wh_phrase_extractor = get_giveme5w1h_extractor()
        self.text_segment_batch_processors = self.get_text_segment_batch_processors()

    def get_text_segment_batch_processors(self):
        text_segments = self.request_params.text_segments
        batch_size = self.request_params.batch_size
        batch_processors = []
        cur_batch_segments = []
        for i, text_segment in enumerate(text_segments):
            cur_batch_segments.append(text_segment)
            if len(cur_batch_segments) >= batch_size or i == len(text_segments) - 1:
                batch_processor = TextSegmentBatchProcessor.from_text_segments_and_extractor( cur_batch_segments, self.wh_phrase_extractor )
                batch_processors.append(batch_processor)
                cur_batch_segments = []
        return batch_processors

    def get_response(self):
        batch_results = []
        max_num_candidate_phrases = self.request_params.max_num_candidate_phrases
        for batch_processor in self.text_segment_batch_processors:
            cur_batch_results = {
                "id's" : batch_processor.get_text_segment_ids(),
                "wh-phrases" : batch_processor.get_wh_phrases( max_num_candidate_phrases=max_num_candidate_phrases )
            }
            batch_results.append(cur_batch_results)
        return jsonify(batch_results)
