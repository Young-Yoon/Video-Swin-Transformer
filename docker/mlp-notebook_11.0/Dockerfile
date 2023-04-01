FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-tensorflow-cuda-full:v1.3.0-rc.1
SHELL ["/bin/bash", "-c"]

# Labels for the docker
LABEL description="This mlp-notebook image has pytorch 1.7.1 in python37 with cuda 11.0.3, and cudnn 8.0." \
      tag="pytorch1.7.1-py37-cuda11.0-openmpi-mmcv-apex-timm" \
      tooltype="pytorch" tooltypeversion="1.7.1"

USER root:root

# libGL1
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt-get update && apt-get install -y libgl1 awscli s3fs

RUN conda install -y gcc_linux-64 && pip install ruamel.yaml
RUN pip uninstall -y tensorflow-gpu tensorflow-estimator tensorflow-metadata kubernetes kfp kfp-server-api kfserving

#ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ARG USER_NAME=jovyan
USER $USER_NAME

ARG ENV_NAME=mmlab
RUN conda create -n $ENV_NAME -y python=3.7
ENV CONDABIN="/opt/conda/bin"
ENV CONDAENV="/opt/conda/envs/$ENV_NAME/bin"
#ENV ACTIVATE="$CONDABIN/activate $ENV_NAME"
RUN echo "conda deactivate && conda activate $ENV_NAME" >> /home/$USER_NAME/.bashrc

RUN conda install -n $ENV_NAME -y -c pytorch pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 pillow=6.1
RUN conda install -n $ENV_NAME -y -c "conda-forge/label/cf202003" nvidia-apex
RUN conda install -n $ENV_NAME -y -c anaconda ipykernel==6.15.2
RUN conda install -n $ENV_NAME -y pip
RUN $CONDAENV/pip install mmcv==1.4.0 timm==0.6.12 scipy==1.7.3 matplotlib==3.5.3
RUN $CONDAENV/python -m ipykernel install --user --name=$ENV_NAME
#pip install -U pip && python -m pip install -U matplotlib==3.5.3 && pip install --upgrade setuptools

WORKDIR /home/$USER_NAME
RUN mkdir -p .aws
