
### Setup instruction

#1. Login server, and Docker build: 
	-> cd ~/demo
	-> ./build.sh


#2. Login server, and Docker Run: 
	-> cd ~/demo
	-> ./start.sh

#3. IDE Execution

1. your audio file should be in mp3 format
2. copy your audio file into ./voice folder, and mark down the relative path name of the audio file (e.g. ./voice/2796798189.mp3)
3. modify infer.py if necessary for infer parameters setup
4. edit configuration to set the parameter for run 

	# 1. --transcribe ./voice/2796798189.mp3
	# 2. --train

5. run and wait for the output, 	
	# for transribe, it takes around 10 seconds for 5 mins audio files, enjoy
	# for train, it takes around 60 epoche for 5000 training size (mozilla-foundation/common_voice_11_0 - yue/cantonese samples) at current config

########## Appendix ##########

# Docker Run Command Reference to spin-up docker container

--- configuration	
docker run -itd -u root                                                                    \
--ipc=host --network=host --name whisper                                                   \
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

# Docker Exec Command Reference for login into docker
docker exec -u root -it whisper /bin/bash

# Source Environment Variables, available at set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
source /root/miniconda3/bin/activate whisper