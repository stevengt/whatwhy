#!/usr/bin/env bash

docker swarm init --advertise-addr 127.0.0.1 # $(hostname -I | awk '{print $1}')
docker-compose up -d
docker stack deploy -c docker-compose.yml giveme5w1h-server-cluster
