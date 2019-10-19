FROM ubuntu:latest
WORKDIR  /opt
ENV LC_ALL C.UTF-8
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        git \
        openjdk-8-jdk \
        python3 \
        python3-pip \
        unzip \
        wget
RUN pip3 install giveme5w1h
RUN giveme5w1h-corenlp install
EXPOSE 9099
COPY start-giveme5w1h-rest-server.sh start-giveme5w1h-rest-server.sh
CMD "/opt/start-giveme5w1h-rest-server.sh"