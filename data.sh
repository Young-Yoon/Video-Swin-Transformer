#!/bin/bash

dhome="$HOME"
dswin=$HOME/swin-data
dpath="$dhome/data"
dname=kinetics400
dgz=k400models.tar.gz
drepo=Video-Swin-Transformer

if [[ "1" == "0" ]]; then
aws s3 ls

[[ -e "$dpath" ]] || mkdir -p $dpath
aws s3 cp s3://$dname/$dgz $dpath 
tar xvzf $dpath/$dgz -C $dpath/
rm $dpath/$dgz

echo link dataset
[[ -e "$dswin/$drepo/data" ]] || ln -s $dpath $dswin/$drepo/data

if [[ "${CONDA_PREFIX##*/}" != "mmlab" ]]; then
	source $dswin/anaconda3/bin/activate mmlab
fi

pushd $dswin/$drepo
for mdl in tiny small base; do
	[[ "$mdl" == "base" ]] && trsz="1k 22k" || trsz="1k"
	for tr in $trsz; do 
python tools/test.py \
        ./configs/recognition/swin/swin_${mdl}_patch244_window877_kinetics400_${tr}.py \
        ./data/kinetics400/models/swin_${mdl}_patch244_window877_kinetics400_${tr}.pth \
        --eval top_k_accuracy
	done
done
popd

fi

pushd $dswin/$drepo
mdl=small
tr=1k
python tools/train.py \
	./configs/recognition/swin/swin_${mdl}_patch244_window877_kinetics400_${tr}.py \
	--cfg-options load_from=./data/kinetics400/models/swin_${mdl}_patch244_window877_kinetics400_${tr}.pth
popd

