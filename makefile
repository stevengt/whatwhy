
export WHATWHY_ROOT_DIR=${PWD}
export WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/webservices/wh_phrase_extractor/docker/docker-compose.yml
export WH_PHRASE_EXTRACTOR_DOCKER_FILE=whatwhy/webservices/wh_phrase_extractor/docker/Dockerfile
export OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE=whatwhy/webservices/opinion_word_extractor/docker/docker-compose.yml

.PHONY: build-wh-phrase-extractor \
		build-opinion-word-extractor \
		start-wh-phrase-extractor \
		start-opinion-word-extractor \
		stop-wh-phrase-extractor \
		stop-opinion-word-extractor \
		init-swarm \
		leave-swarm \
		deploy-wh-phrase-extractor-to-swarm \
		remove-wh-phrase-extractor-from-swarm \
		clean

build-wh-phrase-extractor:
	docker-compose --file ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} build --parallel

build-opinion-word-extractor:
	docker-compose --file ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE} build --parallel

start-wh-phrase-extractor:
	docker-compose up ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE}

start-opinion-word-extractor:
	docker-compose up ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE}

stop-wh-phrase-extractor:
	docker-compose down ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE}

stop-opinion-word-extractor:
	docker-compose down ${OPINION_WORD_EXTRACTOR_DOCKER_COMPOSE_FILE}

init-swarm:
	docker swarm init --advertise-addr $$(hostname -I | awk '{print $$1}')

leave-swarm:
	docker swarm leave --force

deploy-wh-phrase-extractor-to-swarm:
	docker stack deploy -c ${WH_PHRASE_EXTRACTOR_DOCKER_COMPOSE_FILE} wh-phrase-extractor-cluster

remove-wh-phrase-extractor-from-swarm:
	docker stack rm wh-phrase-extractor-cluster

clean:
	find . -name "*.py[cod]" -delete
	find . -name "*__pycache__" -delete
	docker container prune
	docker image prune --all

default: build-wh-phrase-extractor
