#!/bin/bash

for i in {4..10}
    do
        echo "evo $i times"
        python tools/evo_server.py --population 100
        echo "evo $i : sleep 4h "
        sleep 4h
    done
echo "evo is done!"