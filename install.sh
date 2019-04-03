#!/bin/bash

ENV_NAME=$1

source activate $ENV_NAME

pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install -r requirements.txt
