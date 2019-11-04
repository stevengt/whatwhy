FROM nlpbox/corenlp:3.8.0
ENV CORENLP_SERVER_COMMAND="java -Xms3g -Xmx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --timeout 15000 --preload \"tokenize,ssplit,pos,lemma,parse,ner,depparse,mention,coref\""
RUN echo "${CORENLP_SERVER_COMMAND} &" > start-server && chmod +x start-server
RUN echo "pkill java" > stop-server && chmod +x stop-server
RUN echo "./stop-server && ./start-server" > restart-server && chmod +x restart-server
RUN echo "./start-server; while true; do sleep 1800; ./restart-server; done" > run-server && chmod +x run-server
CMD ./run-server
