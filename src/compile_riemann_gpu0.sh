#!/bin/bash

TARGET=test_riemann_local_upsample_gpu0
TARGET_DIR=build

CWD=$PWD
SRC_FILE=$CWD/test_riemann_local_upsample_gpu0.rg
INPUT_FILE=$CWD/input_riemann_GPUbig.rg
DATA_DIR=$CWD/data

if [ -f $CWD/input.rg ]; then
    rm $CWD/input.rg
fi
ln -s $INPUT_FILE $CWD/input.rg

if [ ! -f $SRC_FILE ]; then
    echo "[ERROR] Cannot find \"$SRC_FILE\"."
    exit
else
    echo "Compile source code \"$SRC_FILE\" to \"$TARGET_DIR\"."
fi

if [ ! -d $DATA_DIR ]; then
    echo "Create a new directory \"$DATA_DIR\""
    mkdir -p $DATA_DIR
fi

if [ ! -d $TARGET_DIR ]; then
    echo "$TARGET_DIR does not exist."
    mkdir -p $TARGET_DIR && cd $TARGET_DIR
else
    echo "$TARGET_DIR already exists."
    cd $TARGET_DIR
fi

build_option="-fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -findex-launch 1 -ffuture 0 -fgpu-arch pascal"
OBJNAME=$TARGET regent.py $SRC_FILE $build_option \
    && echo "Compile completed."

cd $CWD
