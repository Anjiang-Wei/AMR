#!/bin/bash
module purge
module load
module load slurm cmake cuda/12.1 mpi/openmpi/4.1.5

PACKAGE_ROOT_DIR=/scratch2/songhang/Packages
LEGION_ROOT_DIR=$PACKAGE_ROOT_DIR/legion
TARGET_DIR=$PACKAGE_ROOT_DIR/Regent

export CPLUS_INCLUDE_PATH=$HDF5_ROOT_DIR/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$HDF5_ROOT_DIR/lib:$LD_LIBRARY_PATH

date

which gcc
gcc --version
which g++
g++ --version

CWD=$PWD

if [ ! -d "$TARGET_DIR" ]; then
    echo "$TARGET_DIR does not exist."
    mkdir -p $TARGET_DIR && cd $TARGET_DIR
else
    echo "$TARGET_DIR already exists."
    cd $TARGET_DIR
fi

CC=gcc CXX=g++ USE_CUDA=1 USE_GASNET=1 CONDUIT=ibv USE_HDF=1 $LEGION_ROOT_DIR/language/scripts/setup_env.py --prefix $TARGET_DIR --terra-cmake |& tee log

cd $CWD
