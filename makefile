
export WHATWHY_ROOT_DIR=${PWD}
export WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/webservices/wh_phrase_extractor/docker/docker-compose.yml
export WH_PHRASE_EXTRACTOR_DOCKER_FILE=whatwhy/webservices/wh_phrase_extractor/docker/Dockerfile
export OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/webservices/opinion_word_extractor/docker/docker-compose.yml

.PHONY: build \
		start-webservices \
		stop-webservices \
		clean

build:
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} pull corenlp-service
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} build --parallel
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} pull

start-webservices:
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} up
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} up

stop-webservices:
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} down
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} down

clean:
	find . -name "*.py[cod]" -delete
	find . -name "*__pycache__" -delete
	docker container prune
	docker image prune --all
	docker network prune

default: build
