
export WHATWHY_ROOT_DIR=${PWD}
export WH_PHRASE_EXTRACTOR_DOCKER_FILE=whatwhy/text_processing/batch_processors/wh_phrases/Dockerfile
export WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/text_processing/batch_processors/wh_phrases/docker-compose.yml
export CORE_NLP_SERVER_DOCKER_FILE=whatwhy/text_processing/batch_processors/wh_phrases/corenlp-server/Dockerfile

.PHONY: build \
		run-wh-phrase-extractor \
		clean

build:
	docker build --file ${WH_PHRASE_EXTRACTOR_DOCKER_FILE} \
				 --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
				 --build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
				 --build-arg WHATWHY_SOURCE_TYPE=${WHATWHY_SOURCE_TYPE} \
				 --build-arg WHATWHY_SOURCE_NAME=${WHATWHY_SOURCE_NAME} \
				 --build-arg WHATWHY_DEST_TYPE=${WHATWHY_DEST_TYPE} \
				 --build-arg WHATWHY_DEST_NAME=${WHATWHY_DEST_NAME} \
				 --build-arg WHATWHY_ID_COL_NAME=${WHATWHY_ID_COL_NAME} \
				 --build-arg WHATWHY_SOURCE_COL_NAME=${WHATWHY_SOURCE_COL_NAME} \
				 -t wh-phrase-extractor .
	docker build --file ${CORE_NLP_SERVER_DOCKER_FILE} -t corenlp-server .

run-wh-phrase-extractor:
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} up

clean:
	find . -name "*.py[cod]" -delete
	find . -name "*__pycache__" -delete
	docker container prune
	docker image prune --all
	docker network prune

default: build
