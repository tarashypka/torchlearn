#!/bin/bash

MODULE_DIR=$(realpath $(dirname $0))
PYTHON_VERSION=3.7
DEVICE=CPU

ANACONDA_PATH="missing"
ENV_NAME="missing"
HELP_MSG="Usage: install.sh --conda=/path/to/anaconda --env=env_name"

err() {
    echo
    echo ${1}
    echo
    echo ${HELP_MSG}
    echo
    exit
}

for param in $@; do
  case ${param} in
    --env=*)
      ENV_NAME=${param#*=}
      shift
      ;;
    --conda=*)
      ANACONDA_PATH=${param#*=}
      shift
      ;;
    --help)
      echo ${HELP_MSG}
      exit
  esac
done

if [[ ${ANACONDA_PATH} == "missing" ]]; then
  err "Not found --conda argument!"
fi

if [[ ${ENV_NAME} == "missing" ]]; then
  err "Not found --env argument!"
fi

echo ENV=${ENV_NAME}
echo ANACONDA=${ANACONDA_PATH}

export PATH=${ANACONDA_PATH}/bin:$PATH

ENV_PATH=${ANACONDA_PATH}/envs/${ENV_NAME}
ENV_INSTALLED=0
if [[ ! -d ${ENV_PATH} ]]; then
    echo "Create new environment at ${ENV_PATH} ..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION}
    ENV_INSTALLED=1
fi
source ${ANACONDA_PATH}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

PIP=${ENV_PATH}/bin/pip

echo "Install pysimple dependency ..."
cd ${MODULE_DIR}
rm -rf .cache/pysimple
git clone https://github.com/tarashypka/pysimple.git .cache/pysimple
cd .cache/pysimple
./install.sh --conda=${ANACONDA_PATH} --env=${ENV_NAME}
cd ${MODULE_DIR}
rm -rf .cache/pysimple

echo "Install pytorch dependency on device ${DEVICE} ..."
if [[ ${DEVICE} == "CPU" ]]; then
    ${PIP} install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
    ${PIP} install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
elif [[ ${DEVICE} == "GPU" ]]; then
    ${PIP} install torch torchvision
fi

${PIP} install -r requirements.txt
${PIP} install .

function run_tests
{
    echo "${ENV_PATH}/bin/python -m unittest discover -s ${MODULE_DIR}/tests -t ${MODULE_DIR} -v"
}

if [[ -d ${MODULE_DIR}/tests ]]; then
    TESTS_PASSED=$($(run_tests) 2>&1 | tee /dev/tty | tail -1)
fi

if [[ ${TESTS_PASSED} != "OK" ]]; then
    exit
fi
