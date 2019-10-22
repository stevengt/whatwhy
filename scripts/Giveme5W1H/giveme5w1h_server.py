"""
This is a simple server to wrap some of the functionality of Giveme5W1H.
"""

import argparse
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor

_5W1H_WORDS = ["who", "what", "when", "where", "why", "how"]

app = Flask(__name__)
corenlp_server_url = None

def get_5w1h_extractor():
    extractor_preprocessor = Preprocessor(corenlp_server_url)
    return MasterExtractor(preprocessor=extractor_preprocessor)

class TextSegment():

    def __init__(self, content=None, id=None):
        if content is None:
            raise BadRequest("Required property 'content' not found in text-segment object.")
        self.content = content
        self.id = id

class TextSegmentBatch():

    def __init__(self, text_segments):
        self.text_segments = text_segments

    def get_text_segment_ids(self):
        ids = [ text_segment.id for text_segment in self.text_segments ]
        return list( filter( None, ids ) )

    def get_aggregate_text_segment_contents(self):
        contents = [ text_segment.content for text_segment in self.text_segments ]
        return ". ".join(contents)

class TextSegmentBatchProcessor():

    def __init__(self, text_segment_batch, extractor):
        self.text_segment_batch = text_segment_batch
        self._5w1h_extractor = extractor
        self._5w1h_phrases = None

    @staticmethod
    def from_text_segments_and_extractor(text_segments, extractor):
        text_segment_batch = TextSegmentBatch(text_segments)
        return TextSegmentBatchProcessor(text_segment_batch, extractor)

    def get_text_segment_ids(self):
        return self.text_segment_batch.get_text_segment_ids()

    def get_5w1h_phrases(self, max_num_candidate_phrases=1):
        if self._5w1h_phrases is None:
            self._5w1h_phrases = {}
            aggregate_text_segment_contents = self.text_segment_batch.get_aggregate_text_segment_contents()
            document = Document(aggregate_text_segment_contents)
            document = self._5w1h_extractor.parse(document)
            for question_type in _5W1H_WORDS:
                candidate_phrases = document.get_answers(question_type)[:max_num_candidate_phrases]
                self._5w1h_phrases[question_type] = [ candidate_phrase.get_parts_as_text() for candidate_phrase in candidate_phrases ]
        return self._5w1h_phrases

class RequestParams():

    @staticmethod
    def get_current_request_params():
        request_params_json = request.get_json(force=True)
        request_params = RequestParams()

        request_params.batch_size = request_params_json.get("batch-size")

        request_params.max_num_candidate_phrases = request_params_json.get("max-num-candidate-phrases")
        if request_params.max_num_candidate_phrases is None:
            request_params.max_num_candidate_phrases = 1

        text_segments = request_params_json.get("text-segments")
        if text_segments is None or len(text_segments) < 1:
            raise BadRequest("Required property 'text-segments' is missing.")
        request_params.text_segments = [ TextSegment(**text_segment) for text_segment in text_segments ]

        return request_params

class RequestHandler():
    def __init__(self):
        self.request_params = RequestParams.get_current_request_params()

class Get5W1HPhrasesRequestHandler(RequestHandler):

    def __init__(self):
        super().__init__()
        extractor_preprocessor = Preprocessor(corenlp_server_url)
        self._5w1h_extractor = MasterExtractor(preprocessor=extractor_preprocessor)
        self.text_segment_batch_processors = self.get_text_segment_batch_processors()

    def get_text_segment_batch_processors(self):
        text_segments = self.request_params.text_segments
        batch_size = self.request_params.batch_size
        batch_processors = []
        cur_batch_segments = []
        for i, text_segment in enumerate(text_segments):
            cur_batch_segments.append(text_segment)
            if len(cur_batch_segments) >= batch_size or i == len(text_segments) - 1:
                batch_processor = TextSegmentBatchProcessor.from_text_segments_and_extractor( cur_batch_segments, self._5w1h_extractor )
                batch_processors.append(batch_processor)
                cur_batch_segments = []
        return batch_processors

    def get_response(self):
        batch_results = []
        max_num_candidate_phrases = self.request_params.max_num_candidate_phrases
        for batch_processor in self.text_segment_batch_processors:
            cur_batch_results = {
                "id's" : batch_processor.get_text_segment_ids(),
                "5w1h-phrases" : batch_processor.get_5w1h_phrases( max_num_candidate_phrases=max_num_candidate_phrases )
            }
            batch_results.append(cur_batch_results)
        return jsonify(batch_results)

@app.route('/get5w1h-phrases', methods=['POST'])
def get_5w1h_phrases():
    return Get5W1HPhrasesRequestHandler().get_response()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip-address", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--core-nlp-server-host", default="localhost", help="DNS name or IP address of core-nlp server without HTTP prefix.")
    parser.add_argument("--core-nlp-server-port", type=int, default=9000)
    args = parser.parse_args()
    
    print(f"Starting server on {args.ip_address}:{args.port}")

    global corenlp_server_url
    corenlp_server_url = "http://" + str(args.core_nlp_server_host) + ":" + str(args.core_nlp_server_port)
    print(f"Using Stanford Core NLP server at {corenlp_server_url}")

    app.run(args.ip_address, args.port, args.debug)
    print("Server has stopped.")

if __name__ == "__main__":
    main()
