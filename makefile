
export WHATWHY_ROOT_DIR=${PWD}
export WH_PHRASE_EXTRACTOR_DOCKER_FILE=whatwhy/text_processing/wh_phrases/Dockerfile
export OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/webservices/opinion_word_extractor/docker/docker-compose.yml

.PHONY: build \
		run-wh-phrase-extractor \
		start-webservices \
		stop-webservices \
		clean

build:
	docker build --file ${WH_PHRASE_EXTRACTOR_DOCKER_FILE} -t wh-phrase-extractor
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} pull

run-wh-phrase-extractor:
	docker run wh-phrase-extractor

start-webservices:
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} up -d

stop-webservices:
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} down

clean:
	find . -name "*.py[cod]" -delete
	find . -name "*__pycache__" -delete
	docker container prune
	docker image prune --all
	docker network prune

default: build
