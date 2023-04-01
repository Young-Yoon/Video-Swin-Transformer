#!/bin/bash

dhome="$HOME"
droot=$HOME
dconda=$HOME  # $droot
dpath="$dhome/data"   # path to dataset
dname=kinetics400
dgz=k400models.tar.gz
dname2=xd-violence
dgz2=xdv_test12.tar.gz
dsrc=${PWD##*/}  #Video-Swin-Transformer # repo name
envname=mmlab

while getopts c: flag
do
    case "${flag}" in
        c) oconda=${OPTARG};;
    esac
done

aws s3 ls || exit 1

if [ ! -e "$dpath/$dname/val/hyt69aadDDU_000120_000130.mp4" ]; then
echo prepare $dname validation dataset and models
aws s3 cp s3://$dname/$dgz $dpath
pushd $dpath
tar xvzf $dgz -C . && rm $dgz
popd
fi

if [ ! -e "$dpath/$dname2/test12/v=yDqThVpu1AM__#1_label_B4-0-0.mp4" ]; then
echo prepare $dname2 test dataset
[[ -e "$dpath/$dname2" ]] || mkdir -p $dpath/$dname2
aws s3 cp s3://$dname2/$dgz2 $dpath
pushd $dpath
tar xvzf $dgz2 -C $dname2 && rm $dgz2
popd
fi

echo link dataset
[[ -e "$droot/$dsrc/data" ]] || ln -s $dpath $droot/$dsrc/data
ls -al $droot/$dsrc/data

if [ "$oconda" == "y" ] || [[ "${CONDA_DEFAULT_ENV}" != "${envname}" ]]; then   # "${CONDA_PREFIX##*/}"
	# source $dconda/anaconda3/bin/activate mmlab
	source ~/.bashrc
	conda deactivate
	conda activate $envname
	echo changed to conda env: ${CONDA_DEFAULT_ENV}
fi

echo current conda env: "${CONDA_DEFAULT_ENV}"
pushd $droot/$dsrc

echo "checking setup mmaction2"
ommact=`echo $(conda list | grep mmaction2 | wc -l) | sed -e 's/^[[:space:]]*//'`
[[ "$ommact" != "1" ]] && python setup.py develop

echo "setup jupyter ipykernel"
[[ -e $HOME/.local/share/jupyter/kernels/$envname ]] || python -m ipykernel install --user --name=$envname

echo "inferencing the pretrained model on $dname"
for mdl in tiny ; do # small base; do
	[[ "$mdl" == "base" ]] && trsz="1k 22k" || trsz="1k"
	for tr in $trsz; do 
python tools/test.py \
        ./configs/recognition/swin/swin_${mdl}_patch244_window877_kinetics400_${tr}.py \
        ./data/kinetics400/models/swin_${mdl}_patch244_window877_kinetics400_${tr}.pth \
        --eval top_k_accuracy
	done
done
popd 

pushd $droot/$dsrc
mdl=base
tr=1k
python tools/train.py \
	./configs/recognition/swin/swin_${mdl}_patch244_window877_kinetics400_${tr}.py \
	--cfg-options load_from=./data/kinetics400/models/swin_${mdl}_patch244_window877_kinetics400_${tr}.pth
popd

${CONDA_EXE} deactivate
