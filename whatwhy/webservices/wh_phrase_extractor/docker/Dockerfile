FROM python:3.7-slim-buster
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        unzip \
        wget
RUN pip install giveme5w1h && \
    python -m nltk.downloader wordnet
ENV LC_ALL C.UTF-8
EXPOSE 9099 
WORKDIR /opt/whatwhy
COPY . .
RUN pip install .
CMD wh-phrase-extractor-server --core-nlp-server-host $CORE_NLP_SERVER_HOST
