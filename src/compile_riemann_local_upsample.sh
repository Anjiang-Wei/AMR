#!/bin/bash

TARGET=test_riemann_local_upsample
TARGET_DIR=build

CWD=$PWD
SRC_FILE=$CWD/$TARGET.rg
DATA_DIR=$CWD/data

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


OBJNAME=$TARGET regent.py $SRC_FILE -fflow 0 \
    && echo "Compile completed."

cd $CWD
