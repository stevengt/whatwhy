#!/bin/bash

giveme5w1h-corenlp &
python3 -c "import Giveme5W1H.examples.extracting.server as server; server.host='0.0.0.0'; server.main()"
