"""
This is a server to wrap some of the functionality of Giveme5W1H.

To get the 5W1H ("who", "what", "when", "where", "why", "how") phrases
from text segments, send a POST request to the server endpoint /get-5w1h-phrases
with a JSON body formatted as follows:

    Request Format:
        - "text-segments"   : An array of objects with a "content" field and an optional "id" field.
        - "batch-size"      : (Optional) If specified, this will concatenate consecutive text segments
                              into batches of the requested size to be processed as a single text segment.
                              This may speed up execution time but result in a loss of accuracy.
        - "max-num-candidate-phrases" : (Optional) This specifies the maximum number of phrases to
                                        return for each 5W1H word.

    Response Format:
        - The response will be an array of objects with the following fields,
          each corresponding to the results of one text segment batch:
            - "id's"        : An array of the ID's of all text segments in the batch.
            - "5w1h-phrases": An object with the following fields, each of which contain an array
                              of candidate phrases for the corresponding 5W1H word:
                                -"who", "what", "when", "where", "why", "how"
"""

import argparse
from flask import Flask
from whatwhy.services.giveme5w1h_proxy_server.request_handlers import Get5W1HPhrasesRequestHandler

app = Flask(__name__)
corenlp_server_url = None

@app.route('/get-5w1h-phrases', methods=['POST'])
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
