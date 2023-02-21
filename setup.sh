#!/bin/bash

usage() { echo "Usage: $0 [-t <0|1|2>] [-m <gpu>] [ -a(ws) | -c(onda) | -e(nv) | -l(ab) | -p(ytorch) ]" 1>&2; exit 1; }

device=mlp
taws=0 && tconda=0 && tenv=0 && tmmlab=0 && tpytorch=0
task=1
while getopts t:m:a:c:e:l:p: flag
do
  case "${flag}" in
    t) task=${OPTARG};;    # decimal int
    m) device=${OPTARG};;    # mlp | m60 | t4 | a10
    a) taws=2;;
    c) tconda=2;;
    e) tenv=2;;
    l) tmmlab=2;;
    p) tpytorch=2;;
    *) echo invalid option && usage;;
  esac
done
echo install level: "$task", device: "$device"

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
exit 1

if [ $tenv -ge "$task" ]; then
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
conda install pytorch-gpu cudatoolkit="$vcuda" -c conda-forge  # pytorch-gpu-1.11.0 | cudnn-8.2.1.32
conda install pillow=6.1 -c pytorch
pip install torch=="$vtorch" torchvision=="$vvision" -f https://download.pytorch.org/whl/torch_stable.html
fi

if true; then
echo nvidia apex
conda install -c "conda-forge/label/cf202003" nvidia-apex

echo pip upgrade
python -mpip install -U pip && python -mpip install -U matplotlib && pip install --upgrade setuptools

echo open-mmlab
pip install mmcv==1.4.0 timm scipy

pushd ~/data || exit 1
git clone https://github.com/SwinTransformer/Video-Swin-Transformer 
pushd Video-Swin-Transformer || exit 1
python setup.py develop
popd
popd
fi

