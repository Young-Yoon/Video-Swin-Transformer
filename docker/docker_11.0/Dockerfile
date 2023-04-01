FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]
# Labels for the docker
LABEL description="This docker has pytorch 1.7.1 in python37 with cuda 11.0.3, and cudnn 8.0." \
      tag="pytorch1.7.1-py37-cuda11.0-openmpi-mmcv-apex-timm" \
      creator="ninja" tooltype="pytorch" \
      tooltypeversion="1.7.1" \
      createtime="06/08/2021"

USER root:root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    wget \
    vim \
    awscli s3fs && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt update && apt upgrade -y && apt install s3fs awscli
#pip install --upgrade pip
#pip install awscli

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

# ssh
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

# add user
ARG UID=1000
ARG USER_NAME=avc
RUN adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

ARG condapath="/home/$USER_NAME/conda"
RUN echo 'export PATH=$condapath/bin:$PATH' > /etc/profile.d/conda.sh

USER $USER_NAME

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p $condapath && \
    rm ~/anaconda.sh

ENV PATH=$condapath/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ARG ENV_NAME=mmlab
RUN conda create -y -n $ENV_NAME python=3.7
ENV CONDAENV="$condapath/envs/$ENV_NAME/bin"
#ENV ACTIVATE="$condapath/bin/activate $ENV_NAME"
RUN conda init bash && echo "conda deactivate && conda activate $ENV_NAME" >> /home/$USER_NAME/.bashrc

RUN conda install -n $ENV_NAME -y -c pytorch pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 pillow=6.1
RUN conda install -n $ENV_NAME -y -c "conda-forge/label/cf202003" nvidia-apex
RUN conda install -n $ENV_NAME -y -c anaconda ipykernel==6.15.2
RUN conda install -n $ENV_NAME -y pip
#RUN $CONDAENV/python -m pip install -U pip && python -m pip install -U matplotlib && pip install --upgrade setuptools
RUN $CONDAENV/pip install mmcv==1.4.0 timm==0.6.12 scipy==1.7.3 matplotlib==3.5.3
RUN $CONDAENV/python -m ipykernel install --user --name=$ENV_NAME

WORKDIR /home/$USER_NAME
#RUN git clone https://github.com/SwinTransformer/Video-Swin-Transformer
#WORKDIR /home/$USER_NAME/Video-Swin-Transformer
#RUN $ACTIVATE && python setup.py develop
RUN . .bashrc
RUN mkdir -p .aws
