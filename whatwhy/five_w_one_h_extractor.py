import logging
import json
import requests

_5W1H_WORDS = ["who", "what", "when", "where", "why", "how"]
REQUEST_TIMEOUT_IN_SECONDS = 5

class FiveWOneHExtractor():
    """
    Class for extracting the 5w1h phrases ('who', 'what', 'when', 'where', 'why', 'how')
    from raw text.

    This class requires access to a Giveme5W1H REST API endpoint (see
    https://github.com/fhamborg/Giveme5W1H for more details). This is assumed to be
    at http://localhost:9099/extract by default.
    """

    def __init__(self, server_url="http://localhost:9099/extract"):
        self.server_url = server_url
    
    def get_5w1h_dict_from_text(self, text):
        _5w1h_dict = {}
        for question_type in _5W1H_WORDS:
            _5w1h_dict[question_type] = ""
        
        try:
            api_request_params = self.get_api_request_params_from_text(text)
            response = requests.post( self.server_url, data=api_request_params, timeout=REQUEST_TIMEOUT_IN_SECONDS )
            if response.ok and response.content is not None:
                response_json = json.loads( response.content, encoding='utf-8', strict=True )
                for question_type in _5W1H_WORDS:
                    _5w1h_dict[question_type] = self.get_top_answer_from_response_json(response_json, question_type)
            else:
                logging.warning(response)
        except Exception as e:
            logging.error(f"Unable to get 5W1H phrases from Giveme5W1H server: {e}")
        
        return _5w1h_dict

    def get_api_request_params_from_text(self, text):
        # Some parameters have placeholder values because they are required by the Giveme5W1H API.
        request_params = {
            "title": text,
            "date": "2000-01-01 12:00:00", 
            "url" : "dummy.url",
            "language": "en"
        }
        return json.dumps(request_params).encode("utf-8")
    
    def get_top_answer_from_response_json(self, response_json, question_type):
        try:
            return response_json["fiveWoneH"][question_type]["extracted"][0]["text"]
        except:
            return ""
