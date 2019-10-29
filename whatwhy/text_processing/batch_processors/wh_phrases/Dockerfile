FROM ubuntu:latest
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        git \
        openjdk-8-jdk \
        python3 \
        python3-pip \
        unzip \
        wget
WORKDIR /tmp
RUN pip3 install giveme5w1h && \
    git clone https://github.com/fhamborg/Giveme5W1H.git && \
    rm -rf /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    mkdir -p /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches && \
    mv ./Giveme5W1H/Giveme5W1H/examples/caches/* /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    rm -rf ./Giveme5W1H && \
    giveme5w1h-corenlp install && \
    python3 -m nltk.downloader wordnet
ENV LC_ALL C.UTF-8
WORKDIR /opt/whatwhy
COPY . .
RUN pip3 install .
CMD wh-phrase-extractor
