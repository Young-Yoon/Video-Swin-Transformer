FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

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

#RUN apt update && apt upgrade -y && apt install s3fs
#pip install --upgrade pip
#pip install awscli


RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

# ssh
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

# add user
ARG UID=1000
ARG USER_NAME=video
RUN adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME
USER $USER_NAME

RUN conda create -y -n mmlab python=3.7

RUN conda install -y python-gpu cudatoolkit=11.0.3 -c conda-forge 
RUN conda install pillow=6.1 -c pytorch -y
RUN conda install -c "conda-forge/label/cf202003" nvidia-apex -y
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install -U pip && python -m pip install -U matplotlib && pi pinstall --upgrade setuptools
RUN pip install mmcv==1.4.0 timm scipy

WORKDIR $HOME
RUN git clone https://github.com/SwinTransformer/Video-Swin-Transformer
RUN cd Video-Swin-Transformer && python setup.py develop

