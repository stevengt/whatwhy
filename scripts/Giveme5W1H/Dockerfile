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
RUN pip3 install giveme5w1h && \
    giveme5w1h-corenlp install && \
    python3 -m nltk.downloader wordnet
ENV LC_ALL C.UTF-8
EXPOSE 9099 
WORKDIR /opt/giveme5w1h
COPY start-giveme5w1h-rest-server.sh start-giveme5w1h-rest-server.sh
CMD /opt/giveme5w1h/start-giveme5w1h-rest-server.sh
