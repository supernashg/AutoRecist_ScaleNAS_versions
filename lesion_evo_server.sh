#!/bin/bash

for i in {1..10}
    do
        echo "evo $i times"
        python tools/evo_server.py --population 100
        echo "evo $i : sleep 5h "
        sleep 5h
    done
echo "evo is done!"