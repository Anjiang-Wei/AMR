#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -p gpu
#SBATCH -t 00:01:00

EXEC_DIR=build_HelloWorld
EXEC_BIN=$EXEC_DIR/HelloWorld

export LD_LIBRARY_PATH=$EXEC_DIR:$LD_LIBRARY_PATH
GASNET_BACKTRACE=1 mpirun --bind-to none $EXEC_BIN
