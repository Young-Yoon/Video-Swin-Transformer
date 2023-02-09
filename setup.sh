#!/bin/bash

if [ $# -ge 1 ] && [ "$1" == "mlp" ]; then
	echo MLP
	nvidia-smi
	
	echo MLP get-key
	sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05

	echo "MLP- libGL1"
	sudo apt-get update && sudo apt-get install libgl1
fi

if [ $# -ge 1 ] && [ "$1" == "m60" ]; then
	vcuda=10.2.89
	vtorch=1.8.1+cu102
	vvision=0.9.1+cu102
else
	vcuda=11.0.3
	vtorch=1.7.0+cu110
	vvision=0.8.1+cu110
fi

ilevel=0  # 0: 
while getopts i: flag
do
    case "${flag}" in
	    i) ilevel=$((10#${OPTARG}));;
    esac
done
echo install level: $ilevel


if [ $ilevel -ge 2 ]; then
echo s3fs + awscli
sudo apt update && sudo apt upgrade -y
sudo apt install s3fs
pip install --upgrade pip
pip install awscli
fi

if [ $ilevel -ge 2 ]; then
echo conda install
f=Anaconda3-2022.10-Linux-x86_64.sh
[[ -e $f ]] || wget https://repo.anaconda.com/archive/$f
bash $f  # NOT sudo
rm $f
fi

if [ $ilevel -ge 0 ]; then
echo conda_exe: ${CONDA_EXE%/*}
source "${CONDA_EXE%/*}/activate"

conda init bash
conda deactivate
fi

if [ $ilevel -ge 1 ]; then
echo creating conda env: mmlab
conda create -n mmlab python=3.7 -y
fi


if [[ "${CONDA_DEFAULT_ENV}" != "mmlab" ]]; then 
source ${CONDA_EXE%/*}/activate mmlab
echo changed to conda env: ${CONDA_DEFAULT_ENV}
fi
echo current conda env: ${CONDA_DEFAULT_ENV}

if [ $ilevel -ge 0 ]; then
echo cuda:$vcuda / torch:$vtorch / torchvision:$vvision
conda install pytorch-gpu cudatoolkit="$vcuda" -c conda-forge  # pytorch-gpu-1.11.0 | cudnn-8.2.1.32
conda install pillow=6.1 -c pytorch
pip install torch=="$vtorch" torchvision=="$vvision" -f https://download.pytorch.org/whl/torch_stable.html
fi

if [ $ilevel -ge 1 ]; then
echo nvidia apex
conda install -c "conda-forge/label/cf202003" nvidia-apex

echo pip upgrade
python -mpip install -U pip && python -mpip install -U matplotlib && pip install --upgrade setuptools

echo open-mmlab
pip install mmcv==1.4.0 timm scipy

pushd ~/swin-data
git clone https://github.com/SwinTransformer/Video-Swin-Transformer 
pushd Video-Swin-Transformer
python setup.py develop
popd
popd
fi
