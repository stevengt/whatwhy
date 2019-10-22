from werkzeug.exceptions import BadRequest
from Giveme5W1H.extractor.document import Document
from whatwhy import QUESTION_WORDS


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
        self.wh_phrase_extractor = extractor
        self.wh_phrases = None

    @staticmethod
    def from_text_segments_and_extractor(text_segments, extractor):
        text_segment_batch = TextSegmentBatch(text_segments)
        return TextSegmentBatchProcessor(text_segment_batch, extractor)

    def get_text_segment_ids(self):
        return self.text_segment_batch.get_text_segment_ids()

    def get_wh_phrases(self, max_num_candidate_phrases=1):
        if self.wh_phrases is None:
            self.wh_phrases = {}
            aggregate_text_segment_contents = self.text_segment_batch.get_aggregate_text_segment_contents()
            document = Document(aggregate_text_segment_contents)
            document = self.wh_phrase_extractor.parse(document)
            for question_type in QUESTION_WORDS:
                candidate_phrases = document.get_answers(question_type)[:max_num_candidate_phrases]
                self.wh_phrases[question_type] = [ candidate_phrase.get_parts_as_text() for candidate_phrase in candidate_phrases ]
        return self.wh_phrases
