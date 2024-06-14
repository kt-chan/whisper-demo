https://github.com/notepad-plus-plus/notepad-plus-plus/releases/download/v8.6.7/npp.8.6.7.Installer.x64.exe

https://nf-sycdn.kuwo.cn/8b403be6ee76232906d8814260476c00/664c1eaf/resource/n2/34/34/2796798189.mp3?from=vip

https://pypi.tuna.tsinghua.edu.cn/simple/

https://stackoverflow.com/questions/4409502/directory-transfers-with-paramiko

下载地址：CANN 7.0.1.1 CANN-Toolkit 

    - 下载：Ascend-cann-toolkit_7.0.1.1_linux-aarch64.run (https://support.huawei.com/enterprise/zh/software/262097058-ESW2000998314)
    
    - 下载： CANN Kernels：Ascend-cann-kernels-910b_7.0.1.1_linux.run (https://support.huawei.com/enterprise/zh/software/262097058-ESW2000998337)
    
    - 下载：torch_npu-2.1.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl （https://gitee.com/ascend/pytorch/releases/download/v5.0.1.1-pytorch2.1.0/torch_npu-2.1.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl）


将这三个下载的文件放在 /mnt/remote/whisper/ 目录下，稍后会需要挂载到容器中使用。


## DockerFile
/etc/docker/daemon.json 

{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://docker.nju.edu.cn"
  ]
}

# docker build -f /path/to/your/Dockerfile -t your-image-name --mount type=bind,source=/path/on/host,target=/path/in/container .

FROM https://docker.mirrors.ustc.edu.cn/library/ubuntu:22.04
RUN sed -i 's/ports.ubuntu.com/mirror.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get clean  
RUN apt-get update -y 
RUN apt-get install -y net-tools vim wget curl ssh openjdk-8-jdk sudo 
RUN apt-get install -y git gcc g++ make cmake build-esential ffmpeg
RUN chmod +x /mnt/remote/whisper/*
RUN /mnt/remote/whisper/Ascend-cann-toolkit_7.0.1.1_linux-aarch64.run --quiet --full
RUN /mnt/remote/whisper/Ascend-cann-kernels-910b_7.0.1.1_linux.run --quiet --install

#安装 PyTorch 2.1.0 和配套的 PyTorch Adapter（torch_npu）:
 
pip install torch==2.1.0 -i https://mirrors.aliyun.com/pypi/simple/

pip install torch_npu-2.1.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

#安装 Whipser 依赖的包： 

pip install decorator attrs psutil absl-py cloudpickle scipy synr==0.5.0 tornado numpy pandas sentencepiece accelerate transformers==4.37.0 datasets -i https://mirrors.aliyun.com/pypi/simple/



# Execution

docker run -itd -u root \
--ipc=host --network=host --name whisper \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
-v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/vnpu.cfg:/etc/vnpu.cfg \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools \
-v /mnt/remote:/mnt/remote \
ubuntu:whisper /bin/bash



#进入容器：
 
docker exec -it whisper /bin/bash

#执行下面的命令配置环境变量：
 
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH


# Install minicoda3

https://www.rosehosting.com/blog/how-to-install-miniconda-on-ubuntu-22-04/

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh

bash /opt/miniconda-installer.sh

source /root/miniconda3/bin/activate


# 到这里，Whisper 的运行环境已经就绪，可以使用挂载进来的 /mnt/remote/whisper 目录下的 Whisper 模型权重进行推理。





