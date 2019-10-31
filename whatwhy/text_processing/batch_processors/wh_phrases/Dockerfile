FROM python:3.7-slim-buster
WORKDIR /tmp
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        procps \
        swig3.0 \
        unzip \
        wget

RUN pip install giveme5w1h
RUN git clone https://github.com/fhamborg/Giveme5W1H.git && \
    rm -rf /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    mkdir -p /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches && \
    mv ./Giveme5W1H/Giveme5W1H/examples/caches/* /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    rm -rf ./Giveme5W1H
RUN python -m nltk.downloader wordnet

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
RUN pip install .

RUN echo "whatwhy --process wh-phrases \
                  --source-type sqs \
                  --source-name whatwhy-wh-phrase-extraction \
                  --dest-type s3 \
                  --dest-name whatwhy-data/wh-phrases &" > start-whatwhy && \
    chmod +x start-whatwhy
RUN echo "pkill whatwhy" > stop-whatwhy && chmod +x stop-whatwhy
RUN echo "./stop-whatwhy && ./start-whatwhy" > restart-whatwhy && chmod +x restart-whatwhy
RUN echo "./start-whatwhy; while true; do sleep 900; ./restart-whatwhy; done" > run-whatwhy && chmod +x run-whatwhy
CMD ./run-whatwhy
