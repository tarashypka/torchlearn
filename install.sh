#!/bin/bash

PROJ_DIR=$(dirname $0)
cd $PROJ_DIR

ANACONDA=/usr/local/anaconda
export PATH=$ANACONDA/bin:$PATH
ENV_NAME=$1
if [[ $ENV_NAME == "" ]]; then
    ENV_PATH=$ANACONDA
else
    ENV_PATH=$ANACONDA/envs/$ENV_NAME
    if [[ ! -d $ENV_PATH ]]; then
        conda create --prefix=$ENV_PATH --file=conda_libs.txt
    fi
    source activate $ENV_NAME
fi

pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install -r requirements.txt
pip install .
