FROM python:3.6-slim-buster
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

# Install GiveMe5W1H and its required cache files.
RUN pip install giveme5w1h
RUN git clone https://github.com/fhamborg/Giveme5W1H.git && \
    rm -rf /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    mkdir -p /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches && \
    mv ./Giveme5W1H/Giveme5W1H/examples/caches/* /usr/local/lib/python3.7/site-packages/Giveme5W1H/examples/caches/ && \
    rm -rf ./Giveme5W1H

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG WHATWHY_SOURCE_TYPE
ARG WHATWHY_SOURCE_NAME
ARG WHATWHY_DEST_TYPE
ARG WHATWHY_DEST_NAME
ARG WHATWHY_ID_COL_NAME
ARG WHATWHY_SOURCE_COL_NAME

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV WHATWHY_SOURCE_TYPE=${WHATWHY_SOURCE_TYPE}
ENV WHATWHY_SOURCE_NAME="${WHATWHY_SOURCE_NAME}"
ENV WHATWHY_DEST_TYPE=${WHATWHY_DEST_TYPE}
ENV WHATWHY_DEST_NAME="${WHATWHY_DEST_NAME}"
ENV WHATWHY_ID_COL_NAME="${WHATWHY_ID_COL_NAME}"
ENV WHATWHY_SOURCE_COL_NAME="${WHATWHY_SOURCE_COL_NAME}"

# Install whatwhy.
WORKDIR /opt/whatwhy
COPY . .
RUN pip install .

RUN echo "whatwhy-text --process      wh-phrases \
                       --source-type  $WHATWHY_SOURCE_TYPE \
                       --source-name '$WHATWHY_SOURCE_NAME' \
                       --dest-type    $WHATWHY_DEST_TYPE \
                       --dest-name   '$WHATWHY_DEST_NAME' \
                       --id-col      '$WHATWHY_ID_COL_NAME' \
                       --source-col  '$WHATWHY_SOURCE_COL_NAME' &" > start-whatwhy && \
    chmod +x start-whatwhy
RUN echo "pkill whatwhy-text" > stop-whatwhy && chmod +x stop-whatwhy
RUN echo "./stop-whatwhy && ./start-whatwhy" > restart-whatwhy && chmod +x restart-whatwhy

# Restart the process every 30 minutes to prevent it from getting stuck.
RUN echo "./start-whatwhy; while true; do sleep 1800; ./restart-whatwhy; done" > run-whatwhy && chmod +x run-whatwhy

CMD ./run-whatwhy
