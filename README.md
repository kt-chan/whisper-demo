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
  "registry-mirrors": ["https://mirrors.tuna.tsinghua.edu.cn/docker-ce"]
}

# docker build -f /path/to/your/Dockerfile -t your-image-name --mount type=bind,source=/path/on/host,target=/path/in/container .

FROM ubuntu:22.04


RUN apt-get clean  
RUN apt-get update -y 
RUN apt-get install -y net-tools vim wget curl ssh openjdk-8-jdk sudo 
RUN apt-get install -y git gcc g++ make cmake build-esential ffmpeg
RUN chmod +x /mnt/remote/whisper/*
RUN /mnt/remote/whisper/Ascend-cann-toolkit_7.0.1.1_linux-aarch64.run --quiet --full
RUN /mnt/remote/whisper/Ascend-cann-kernels-910b_7.0.1.1_linux.run --quiet --install
