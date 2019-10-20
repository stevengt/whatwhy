"""
This is a simple server to wrap some of the functionality of Giveme5W1H.
This file was adapted from the following example file:
https://github.com/fhamborg/Giveme5W1H/blob/master/Giveme5W1H/examples/extracting/server.py
"""

import argparse
from flask import Flask, request, jsonify
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.tools.file.reader import Reader
from Giveme5W1H.extractor.tools.file.writer import Writer
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor

app = Flask(__name__)
corenlp_server_url = None
reader = Reader()
writer = Writer()

def request_to_document():
    data = request.get_json(force=True)
    document = reader.parse_newsplease(data, 'Server')
    return document

@app.route('/extract', methods=['POST'])
def extract():
    document = request_to_document()
    if document:
        extractor_preprocessor = Preprocessor(corenlp_server_url)
        extractor = MasterExtractor(preprocessor=extractor_preprocessor)
        extractor.parse(document)
        answer = writer.generate_json(document)
        return jsonify(answer)

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
