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

while getopts d:c: flag
do
    case "${flag}" in
        d) odata=${OPTARG};;
        c) oconda=${OPTARG};;
    esac
done

aws s3 ls || exit 1

if [ "$odata" == "y" ]; then
echo prepare validation dataset and models
[[ -e "$dpath/$dname2" ]] || mkdir -p $dpath/$dname2
aws s3 cp s3://$dname2/$dgz2 $dpath
tar xvzf $dpath/$dgz2 -C $dpath/$dname2
aws s3 cp s3://$dname/$dgz $dpath 
tar xvzf $dpath/$dgz -C $dpath/
rm $dpath/$dgz $dpath/$dgz2

echo link dataset
[[ -e "$droot/$dsrc/data" ]] || ln -s $dpath $droot/$dsrc/data
ls -al $droot/$dsrc/data
fi

if [ "$oconda" == "y" ] || [[ "${CONDA_DEFAULT_ENV}" != "mmlab" ]]; then   # "${CONDA_PREFIX##*/}"
	# source $dconda/anaconda3/bin/activate mmlab
	source ~/.bashrc
	conda deactivate
	conda activate mmlab
	echo changed to conda env: ${CONDA_DEFAULT_ENV}
fi

echo current conda env: "${CONDA_DEFAULT_ENV}"
pushd $droot/$dsrc

ommact=`echo $(conda list | grep mmaction2 | wc -l) | sed -e 's/^[[:space:]]*//'`
echo $ommact
[[ "$ommact" != "1" ]] && python setup.py develop

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
