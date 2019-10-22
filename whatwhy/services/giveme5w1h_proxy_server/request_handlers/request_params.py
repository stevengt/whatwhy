from flask import request
from werkzeug.exceptions import BadRequest
from whatwhy.services.giveme5w1h_proxy_server.text_processing import TextSegment

class RequestParams():

    @classmethod
    def get_current_request_params(cls, request_type=None):
        instance = cls()
        params_dict = request.get_json(force=True)
        if request_type == "get-5w1h-phrases":
            instance.initialize_get_5w1h_phrases_request_params(params_dict)
        return instance

    def initialize_get_5w1h_phrases_request_params(self, params_dict):
        batch_size = params_dict.get("batch-size")
        self.batch_size = batch_size if batch_size is not None else 1

        max_num_candidate_phrases = params_dict.get("max-num-candidate-phrases")
        self.max_num_candidate_phrases = max_num_candidate_phrases if max_num_candidate_phrases is not None else 1

        text_segments = params_dict.get("text-segments")
        if text_segments is None or len(text_segments) < 1:
            raise BadRequest("Required property 'text-segments' is missing.")
        self.text_segments = [ TextSegment(**text_segment) for text_segment in text_segments ]
