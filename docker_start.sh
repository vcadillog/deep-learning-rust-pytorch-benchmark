#!/bin/bash

docker build -t python -f images/python/Dockerfile .
docker build -t rust -f images/rust/Dockerfile .

docker run --name python python
docker run --name rust rust

