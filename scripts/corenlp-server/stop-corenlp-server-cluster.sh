#!/usr/bin/env bash

docker stack rm corenlp-server-cluster
docker-compose down
docker swarm leave --force
