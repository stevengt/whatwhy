FROM python:3.7-slim-buster
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        git \
        unzip \
        wget
WORKDIR /tmp
RUN pip install giveme5w1h && \
    git clone https://github.com/fhamborg/Giveme5W1H.git && \
    rm -rf /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    mkdir -p /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches && \
    mv ./Giveme5W1H/Giveme5W1H/examples/caches/* /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    rm -rf ./Giveme5W1H && \
    python -m nltk.downloader wordnet
ENV LC_ALL C.UTF-8
EXPOSE 9099 
WORKDIR /opt/whatwhy
COPY . .
RUN pip install .
CMD wh-phrase-extractor-server --core-nlp-server-host $CORE_NLP_SERVER_HOST
