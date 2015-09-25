#!/bin/bash

echo "Executing"
alias python=/usr/bin/python
export PYTHONPATH=~/dbhw/caffe/python:$PYTHONPATH
/usr/bin/python rasterize_jon_scratch.py /home/vj/dbsh/code/model5_32/snapshot_iter_282285.caffemodel /home/vj/dbsh/code/model5_32/deploy.prototxt  --mean /home/vj/dbsh/code/model5_32/mean.binaryproto --labels /home/vj/dbsh/code/model5_32/labels.txt --nogpu $1