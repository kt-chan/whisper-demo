#!/bin/bash

# Name of the Docker container to check and remove if running
CONTAINER_NAME="whisper"

# Check if the container is running
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME})" ]; then
    # Container is running, remove it with force
    echo "Container '${CONTAINER_NAME}' is running. Removing it..."
    docker rm -f "$CONTAINER_NAME"
fi

docker run -itd -u root                                                                    \
--ipc=host --network=host --name ${CONTAINER_NAME}                                         \
--device=/dev/davinci0                                                                     \
--device=/dev/davinci_manager                                                              \
--device=/dev/devmm_svm                                                                    \
--device=/dev/hisi_hdc                                                                     \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi                                           \
-v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common             \
-v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver             \
-v /etc/ascend_install.info:/etc/ascend_install.info                                       \
-v /etc/vnpu.cfg:/etc/vnpu.cfg                                                             \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info             \
-v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools                           \
-v /mnt/remote/models/whisper/whisper-large-v3:/mnt/remote/models/whisper/whisper-large-v3 \
-v /root/demo:/root/demo								                                   \
whisper /bin/bash
