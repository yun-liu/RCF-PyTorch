#!/bin/bash

rsync -aq ./ ${TMPDIR}

cd $TMPDIR

mkdir data

tar -I pigz -xf /cluster/work/cvl/liuyun/Edges/HED-BSDS.tar.gz -C ${TMPDIR}/data

python train.py --save-dir /cluster/home/liuyun/data/Models/RCF_Caffe
