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
WORKDIR /opt/giveme5w1h
COPY start-giveme5w1h-server.sh start-giveme5w1h-server.sh
COPY giveme5w1h_server.py giveme5w1h_server.py
CMD /opt/giveme5w1h/start-giveme5w1h-server.sh
