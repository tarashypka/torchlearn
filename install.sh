#!/bin/bash

ENV_NAME=$1
MODULE_DIR=$(dirname $0)
PYTHON_VERSION=3.7
ANACONDA_PATH=${HOME}/miniconda3

export PATH=${ANACONDA_PATH}/bin:$PATH

if [[ ${ENV_NAME} == "" ]]; then
    ENV_PATH=${ANACONDA_PATH}
else
    ENV_PATH=${ANACONDA_PATH}/envs/${ENV_NAME}
    if [[ ! -d ${ENV_PATH} ]]; then
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION}
    fi
    source ${ANACONDA_PATH}/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
fi

PIP=${ENV_PATH}/bin/pip
# CPU
${PIP} install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
${PIP} install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
# GPU
#${PIP} install torch torchvision
${PIP} install -r requirements.txt
${PIP} install .
