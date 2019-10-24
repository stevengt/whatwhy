import json
from werkzeug.exceptions import BadRequest
from Giveme5W1H.extractor.document import Document
from whatwhy import QUESTION_WORDS


class TextSegment():

    def __init__(self, content=None, id=None):
        if content is None:
            raise BadRequest("Required property 'content' not found in text-segment object.")
        self.content = content
        self.id = id

    @staticmethod
    def from_json(json_string):
        json_fields = json.loads( json_string, encoding="utf-8", strict=True )
        return TextSegment(**json_fields)

    def to_json(self):
        json_fields = self.as_dict()
        return json.dumps(json_fields)

    def as_dict(self):
        return {
            "content" : self.content,
            "id" : self.id
        }

class TextSegmentBatch():

    def __init__(self, text_segments):
        self.text_segments = text_segments

    def get_text_segment_ids(self):
        ids = [ text_segment.id for text_segment in self.text_segments ]
        return list( filter( None, ids ) )

    def get_aggregate_text_segment_contents(self):
        contents = [ text_segment.content for text_segment in self.text_segments ]
        return ". ".join(contents)

class TextSegmentBatchResults():

    def __init__(self, text_segment_ids, wh_phrases ):
        self.text_segment_ids = text_segment_ids
        self.wh_phrases = wh_phrases

    @staticmethod
    def from_json(json_string):
        json_fields = json.loads( json_string, encoding="utf-8", strict=True )
        return TextSegmentBatchResults(json_fields["id's"], json_fields["wh-phrases"])

    def to_json(self):
        json_fields = self.as_dict()
        return json.dumps(json_fields)

    def as_dict(self):
        return {
            "id's" : self.text_segment_ids,
            "wh-phrases" : self.wh_phrases
        }

class TextSegmentBatchProcessor():

    def __init__(self, text_segments=None, text_segment_batch=None, extractor=None, max_num_candidate_phrases=1):
        """
        Constructor method which accepts a Giveme5W1H extractor and EITHER 
        a list of TextSegment's or a TextSegmentBatch.
        """
        self.text_segment_batch = text_segment_batch if text_segment_batch is not None else TextSegmentBatch(text_segments)
        self.wh_phrase_extractor = extractor
        self.max_num_candidate_phrases = max_num_candidate_phrases
        self.wh_phrases = None

    def get_text_segment_ids(self):
        return self.text_segment_batch.get_text_segment_ids()

    def get_wh_phrases(self):
        if self.wh_phrases is None:
            self.wh_phrases = {}
            aggregate_text_segment_contents = self.text_segment_batch.get_aggregate_text_segment_contents()
            document = Document(aggregate_text_segment_contents)
            document = self.wh_phrase_extractor.parse(document)
            for question_type in QUESTION_WORDS:
                candidate_phrases = document.get_answers(question_type)[:self.max_num_candidate_phrases]
                self.wh_phrases[question_type] = [ candidate_phrase.get_parts_as_text() for candidate_phrase in candidate_phrases ]
        return self.wh_phrases

    def get_results(self):
        text_segment_ids = self.get_text_segment_ids()
        self.wh_phrases = self.get_wh_phrases()
        return TextSegmentBatchResults(text_segment_ids, self.wh_phrases)
