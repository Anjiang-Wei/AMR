#!/bin/bash

TARGET=HelloWorld
TARGET_DIR=build_HelloWorld

CWD=$PWD
SRC_FILE=$CWD/$TARGET.rg

LD_LIBRARY_PATH=$LG_RT_DIR/../bindings/regent/:$LD_LIBRARY_PATH

if [ ! -f $SRC_FILE ]; then
    echo "[ERROR] Cannot find \"$SRC_FILE\"."
    exit
else
    echo "Compile source code \"$SRC_FILE\" to \"$TARGET_DIR\"."
fi

if [ ! -d $TARGET_DIR ]; then
    echo "$TARGET_DIR does not exist."
    mkdir -p $TARGET_DIR && cd $TARGET_DIR
else
    echo "$TARGET_DIR already exists."
    cd $TARGET_DIR
fi


OBJNAME=$TARGET regent.py $SRC_FILE \
    && echo "Compile completed."

cd $CWD
