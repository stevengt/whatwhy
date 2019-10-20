#!/bin/bash

ip_address_of_server_host=$(hostname -I | awk '{print $1}')
python3 giveme5w1h_server.py --core-nlp-server-host $CORE_NLP_SERVER_HOST
                             #--ip-address $ip_address_of_server_host
