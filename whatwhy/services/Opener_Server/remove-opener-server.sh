#!/usr/bin/env bash

# NLP services
docker rm opener-language-identifier
docker rmi cwolff/opener-docker-language-identifier

docker rm opener-tokenizer
docker rmi cwolff/opener-docker-tokenizer

docker rm opener-pos-tagger
docker rmi cwolff/opener-docker-pos-tagger

docker rm opener-constituent-parser
docker rmi devkws/opener-docker-constituent-parser

docker rm opener-ner
docker rmi cwolff/opener-docker-ner

docker rm opener-polarity-tagger
docker rmi devkws/opener-docker-polarity-tagger

docker rm opener-opinion-detector-basic
docker rmi devkws/opener-docker-opinion-detector-basic

# Wrapper service
docker rm opener-wrapper
docker rmi devkws/opener-docker-wrapper

docker network rm opener
