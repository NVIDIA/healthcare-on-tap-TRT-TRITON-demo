#!/bin/bash

#build docker Image
DOCKER_IMAGE="hcot_inference_demo"
docker build -t hcot_inference_demo ./Docker/



GPU_IDs=$1
DATA_DIR=$2
PWD=`pwd`

echo -----------------------------------
echo starting docker for ${DOCKER_IMAGE} using GPUS ${GPU_IDs}
echo -----------------------------------


extraFlag="-it --ipc=host --net=host"
cmd2run="/bin/bash"

echo Please run "./start_jupyter_lab.sh" to enable lab extensions and start the JupyterLab

echo $DOCKER_IMAGE

docker run ${extraFlag} \
  --gpus "device="${GPU_IDs} \
  -v ${PWD}/:/workspace/codes/ \
  -v ${DATA_DIR}:/workspace/data \
  -w /workspace/codes \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}