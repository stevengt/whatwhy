"""
Server for extracting the WH-phrases ('who', 'what', 'when', 'where', 'why', 'how')
from raw text.

The client/server in this package serve as a wrapper around some of the
functionality of Giveme5W1H: https://github.com/fhamborg/Giveme5W1H

Additionally, this package enables "batch" processing by concatenating
multiple text-segments and processing them as a single text-segment.

To get the WH-phrases ("who", "what", "when", "where", "why", "how")
from text segments, send a POST request to the server endpoint /get-wh-phrases
with a JSON body formatted as follows:

    Request Format:
        - "text-segments"   : An array of objects with a "content" field and an optional "id" field.
        - "batch-size"      : (Optional) If specified, this will concatenate consecutive text segments
                              into batches of the requested size to be processed as a single text segment.
                              This may speed up execution time but result in a loss of accuracy.
        - "max-num-candidate-phrases" : (Optional) This specifies the maximum number of phrases to
                                        return for each WH-word.

    Response Format:
        - The response will be an array of objects with the following fields,
          each corresponding to the results of one text segment batch:
            - "id's"        : An array of the ID's of all text segments in the batch.
            - "wh-phrases": An object with the following fields, each of which contain an array
                              of candidate phrases for the corresponding WH-word:
                                -"who", "what", "when", "where", "why", "how"
"""


from .logger import logger
from .request_handlers import GetWHPhrasesRequestHandler
import argparse
from flask import Flask
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from Giveme5W1H.extractor.extractor import MasterExtractor

app = Flask(__name__)

class WHPhraseExtractorServer():

    def __init__( self,
                  ip_address,
                  port,
                  core_nlp_server_host,
                  core_nlp_server_port,
                  should_show_debug_logs ):

        self.ip_address = ip_address
        self.port = port

        if should_show_debug_logs:
            logger.setLevel("DEBUG")
            app.logger.setLevel("DEBUG")

        corenlp_server_url = "http://{host}:{port}".format( host=core_nlp_server_host, port=core_nlp_server_port )
        app.config["corenlp_server_url"] = corenlp_server_url
        logger.info(f"Using Stanford Core NLP server at {corenlp_server_url}")

    extractor_preprocessor = Preprocessor(app.config["corenlp_server_url"])
    app.config["wh_phrase_extractor"] = MasterExtractor(preprocessor=extractor_preprocessor)

    def run(self):
        logger.info(f"Starting server on {self.ip_address}:{self.port}")
        app.run(self.ip_address, self.port)
        logger.info("Server has stopped.")

@app.route('/get-wh-phrases', methods=['POST'])
def get_wh_phrases():
    return GetWHPhrasesRequestHandler().get_response()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip-address", default="0.0.0.0")
    parser.add_argument("--port", default="9099")
    parser.add_argument("--core-nlp-server-host", default="localhost", help="DNS name or IP address of core-nlp server without HTTP prefix.")
    parser.add_argument("--core-nlp-server-port", default="9000")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    
    server = WHPhraseExtractorServer( ip_address=args.ip_address,
                                      port=args.port,
                                      core_nlp_server_host=args.core_nlp_server_host,
                                      core_nlp_server_port=args.core_nlp_server_port,
                                      should_show_debug_logs=args.debug )
    server.run()

if __name__ == "__main__":
    main()
