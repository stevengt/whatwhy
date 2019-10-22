#!/usr/bin/env bash

# NLP services
docker start opener-language-identifier
docker start opener-tokenizer
docker start opener-pos-tagger
docker start opener-constituent-parser
docker start opener-ner
docker start opener-polarity-tagger
docker start opener-opinion-detector-basic

# Wrapper service
docker start opener-wrapper
