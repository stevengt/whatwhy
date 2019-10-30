FROM ubuntu:latest
WORKDIR /tmp
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        git \
        openjdk-8-jdk \
        python3 \
        python3-pip \
        swig3.0 \
        unzip \
        wget

RUN pip3 install giveme5w1h
RUN git clone https://github.com/fhamborg/Giveme5W1H.git && \
    rm -rf /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    mkdir -p /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches && \
    mv ./Giveme5W1H/Giveme5W1H/examples/caches/* /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    rm -rf ./Giveme5W1H
RUN giveme5w1h-corenlp install
RUN python3 -m nltk.downloader wordnet

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

WORKDIR /opt/whatwhy
COPY . .
RUN pip3 install .

CMD whatwhy --process wh-phrases \
            --source-type sqs \
            --source-name whatwhy-wh-phrase-extraction \
            --dest-type s3 \
            --dest-name whatwhy-data/wh-phrases
