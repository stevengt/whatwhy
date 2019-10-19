#!/usr/bin/env bash

docker stack rm giveme5w1h-server-cluster
docker-compose down
docker swarm leave --force
