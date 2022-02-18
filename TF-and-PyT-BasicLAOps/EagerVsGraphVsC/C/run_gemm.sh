#!/bin/bash

#source ${HOME}/.zshrc.lamp
#cd ${LAMP_EXPERIMENTS}/c

#######################################################################
# Environment variables
export CC=gcc
export CXX=g++
#export CFLAGS="-O3 -march=native -fopenmp -m64 -I${MKLROOT}/include"
#export CXXFLAGS="-O3 -march=native -fopenmp -m64 -I${MKLROOT}/include"
#export CPPFLAGS="-O3 -march=native -fopenmp -m64 -I${MKLROOT}/include"

#export LOCAL_INSTALL_DIR=${HOME}/.lamp
#export PATH=${LOCAL_INSTALL_DIR}/bin:${PATH}
#export LD_LIBRARY_PATH=${LOCAL_INSTALL_DIR}:${LD_LIBRARY_PATH}

#######################################################################
# LAMP specific variables
#export LAMP_EXPERIMENTS=${HOME}/exercise/lamp-ml/LAMP_benchmark
#export LAMP_OUTPUT_DIR="${LAMP_EXPERIMENTS}/results/"
export LAMP_L3_CACHE_SIZE="7500000"
export LAMP_REPS=20
export LAMP_N=3000


#export LAMP_N=6000
echo "-----------------------------"
echo "$1 Threads"
echo "-----------------------------"
export MKL_NUM_THREADS=$1
export OMP_NUM_THREADS=$1
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/mac/compiler/lib/:$LD_LIBRARY_PATH"  # Runtime cannot find libiomp5.
#use this path in cluster
#export LD_LIBRARY_PATH="${MKLROOT}/../lib/intel64:$LD_LIBRARY_PATH"  # Runtime cannot find libiomp5.

export LAMP_C_OUTPUT_DIR=results/c_${OMP_NUM_THREADS}.txt

make clean
make all -j 24

echo "algorithm;m;k;n;C;cs_time"                    > ${LAMP_C_OUTPUT_DIR}
./bin/gemm_noup.x  $LAMP_N $LAMP_N $LAMP_N          >> ${LAMP_C_OUTPUT_DIR}

