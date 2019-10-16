#!/usr/bin/env bash

# NLP services
docker stop opener-language-identifier
docker stop opener-tokenizer
docker stop opener-pos-tagger
docker stop opener-constituent-parser
docker stop opener-ner
docker stop opener-polarity-tagger
docker stop opener-opinion-detector-basic

# Wrapper service
docker stop opener-wrapper
