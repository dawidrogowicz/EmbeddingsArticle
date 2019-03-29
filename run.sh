#!/bin/bash
docker run --runtime=nvidia -it -p 0.0.0.0:6006:6006 -u $(id -u):$(id -g) -v $PWD:/home/app/ marian bash -c "cd /home/app; bash"
