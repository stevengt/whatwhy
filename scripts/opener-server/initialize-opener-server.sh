#!/usr/bin/env bash

docker network create opener

# NLP services
docker run -d --net opener --name opener-language-identifier cwolff/opener-docker-language-identifier
docker run -d --net opener --name opener-tokenizer cwolff/opener-docker-tokenizer
docker run -d --net opener --name opener-pos-tagger cwolff/opener-docker-pos-tagger
docker run -d --net opener --name opener-constituent-parser devkws/opener-docker-constituent-parser
docker run -d --net opener --name opener-ner cwolff/opener-docker-ner
docker run -d --net opener --name opener-polarity-tagger devkws/opener-docker-polarity-tagger
docker run -d --net opener --name opener-opinion-detector-basic devkws/opener-docker-opinion-detector-basic

# Wrapper service
docker run -d --net opener --name opener-wrapper -p 9999:80 devkws/opener-docker-wrapper

./stop-opener-server.sh
