#!/bin/bash

usage() { echo "Usage: $0 [-l <0|1|2>] [-m <gpu>] [ -t a(ws)|c(onda)|e(nv)|l(ab)|p(ytorch) ]" 1>&2; exit 1; }

device=mlp
taws=0 && tconda=0 && tenv=0 && tmmlab=0 && tpytorch=0
task=1
while getopts l:m:t: flag
do
  case "${flag}" in
    l) task=${OPTARG};;    # decimal int
    m) device=${OPTARG};;    # mlp | m60 | t4 | a10
    t) taskarg=${OPTARG};; 
    *) echo invalid option && usage;;
  esac
done

[[ "$taskarg" == *"a"* ]] && taws=2
[[ "$taskarg" == *"c"* ]] && tconda=2
[[ "$taskarg" == *"e"* ]] && tenv=2
[[ "$taskarg" == *"l"* ]] && tmmlab=2
[[ "$taskarg" == *"p"* ]] && tpytorch=2

echo install level: "$task", device: "$device", aws$taws, conda$tconda, env$tenv, mmlab$tmmlab, pytorch$tpytorch

if [ "$device" == "mlp" ]; then
  echo MLP
  nvidia-smi

  echo MLP get-key
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05

  echo "MLP- libGL1"
  sudo apt-get update && sudo apt-get install libgl1
fi

if [ "$device" == "m60" ]; then
  vcuda=10.2.89 && vtorch=1.8.1+cu102 && vvision=0.9.1+cu102
else
  vcuda=11.0.3  && vtorch=1.7.0+cu110 && vvision=0.8.1+cu110
fi

if [ $taws -ge "$task" ]; then
echo s3fs + awscli
sudo apt update && sudo apt upgrade -y
sudo apt install s3fs
pip install --upgrade pip
pip install awscli
fi

echo conda_exe: "$CONDA_EXE"
if [ $tconda -ge "$task" ] && [ -z ${CONDA_EXE+x} ]; then   # [[ -z ${var+x} ]] && var none || var exists
echo conda install
f=Anaconda3-2022.10-Linux-x86_64.sh
[[ -e $f ]] || wget https://repo.anaconda.com/archive/$f
bash $f  # NOT sudo
rm $f
fi

if [ $tenv -ge "$task" ]; then
source ~/.bashrc
if [ -z ${CONDA_EXE+x} ]; then echo conda required && exit 1; fi
echo conda_exe: ${CONDA_EXE%/*}
source "${CONDA_EXE%/*}/activate"

conda init bash
conda deactivate
fi

if [ $tmmlab -ge "$task" ]; then
echo creating conda env: mmlab
conda create -n mmlab python=3.7 -y
fi

if [[ "${CONDA_DEFAULT_ENV}" != "mmlab" ]]; then 
source ${CONDA_EXE%/*}/activate mmlab
echo changed to conda env: ${CONDA_DEFAULT_ENV}
fi
echo current conda env: ${CONDA_DEFAULT_ENV}
[[ "${CONDA_DEFAULT_ENV}" != "mmlab" ]] && exit 1

if [ $tpytorch -ge "$task" ]; then
echo cuda:$vcuda / torch:$vtorch / torchvision:$vvision
conda install pytorch-gpu cudatoolkit="$vcuda" -c conda-forge -y  # pytorch-gpu-1.11.0 | cudnn-8.2.1.32
conda install pillow=6.1 -c pytorch -y 
pip install torch=="$vtorch" torchvision=="$vvision" -f https://download.pytorch.org/whl/torch_stable.html
fi

if true; then
echo nvidia apex
conda install -c "conda-forge/label/cf202003" nvidia-apex -y

echo pip upgrade
python -m pip install -U pip && python -m pip install -U matplotlib && pip install --upgrade setuptools

echo open-mmlab
pip install mmcv==1.4.0 timm scipy

pushd ~/dataroot || exit 1
[[ -d Video-Swin-Transformer ]] || git clone https://github.com/SwinTransformer/Video-Swin-Transformer 
pushd Video-Swin-Transformer || exit 1
python setup.py develop
popd
popd
fi

