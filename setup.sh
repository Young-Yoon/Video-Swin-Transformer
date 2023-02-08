#!/bin/bash

if [ $1 == "mlp" ]; then
	echo MLP
	nvidia-smi
	vcuda=11.0.3
	vtorch=1.7.0+cu110
	vvision=0.8.1+cu110
	
	echo MLP get-key
	sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05

	echo "MLP- libGL1"
	sudo apt-get update && sudo apt-get install libgl1
elif [ $1 == "t4" ]
	vcuda=
	vtorch=
	vvision=
fi


echo s3fs + awscli
sudo apt update && sudo apt upgrade -y
sudo apt install s3fs
pip install --upgrade pip
pip install awscli

echo conda install
f=Anaconda3-2022.10-Linux-x86_64.sh
[[ -e $f ]] || wget https://repo.anaconda.com/archive/$f
bash $f  # NOT sudo
rm $f

echo ${CONDA_EXE%/*}
source ${CONDA_EXX%/*}/activate
conda init bash
conda deactivate

echo conda create
conda create -n mmlab python=3.7 -y
conda activate mmlab

echo pip upgrade
python -mpip install -U pip && python -mpip install -U matplotlib && pip install --upgrade setuptools

echo cuda:$v_cuda / torch:$v_torch / torchvision:$v_torchvision
conda install pytorch-gpu cudatoolkit="$vcuda" -c conda-forge  # pytorch-gpu-1.11.0 | cudnn-8.2.1.32
conda install pillow=6.1 -c pytorch
pip install torch=="$vtorch" torchvision=="$vtvision" -f https://download.pytorch.org/whl/torch_stable.html

echo nvidia apex
conda install -c "conda-forge/label/cf202003" nvidia-apex

echo open-mmlab
pip install mmcv==1.4.0 timm scipy

pushd ~/swin-data
git clone https://github.com/SwinTransformer/Video-Swin-Transformer 
pushd Video-Swin-Transformer
python setup.py develop
popd
popd

