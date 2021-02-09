#!/bin/bash

DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:20.11-py3"
GPU_IDs=$1
MODEL_REPO=`pwd`"/"$2

docker run \
  --gpus "device="${GPU_IDs} \
  -v ${MODEL_REPO}/:/models \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  tritonserver --model-repository=/models  --model-control-mode=poll \

#         --log-verbose=1