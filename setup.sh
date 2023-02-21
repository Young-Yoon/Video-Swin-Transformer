#!/bin/bash
# usage: bash setup.sh mlp -i 2  # setup all
device=mlp
taws=0 && tconda=0
task=1
while getopts t:m:c: flag
do
  case "${flag}" in
    t) task=${OPTARG};;    # decimal int
    m) device=${OPTARG};;    # mlp | m60 | t4 | a10
    c) tconda=2;;    # decimal int
  esac
done
echo install level: $ilevel, device: $device

if [ $device == "mlp" ]; then
  echo MLP
  nvidia-smi

  echo MLP get-key
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05

  echo "MLP- libGL1"
  sudo apt-get update && sudo apt-get install libgl1
fi

if [ $device == "m60" ]; then
  vcuda=10.2.89 && vtorch=1.8.1+cu102 && vvision=0.9.1+cu102
else
  vcuda=11.0.3  && vtorch=1.7.0+cu110 && vvision=0.8.1+cu110
fi

if [ $taws -ge $task ]; then
echo s3fs + awscli
sudo apt update && sudo apt upgrade -y
sudo apt install s3fs
pip install --upgrade pip
pip install awscli
fi

if [ $tconda -ge $task -a -z ${CONDA_EXE+x} ]; then
echo conda install
f=Anaconda3-2022.10-Linux-x86_64.sh
[[ -e $f ]] || wget https://repo.anaconda.com/archive/$f
bash $f  # NOT sudo
rm $f
fi
exit 1


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

