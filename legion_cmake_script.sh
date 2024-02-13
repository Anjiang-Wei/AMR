module purge
module load slurm cmake cuda/12.1 mpi/openmpi/4.1.5

PACKAGE_ROOT_DIR=/scratch2/songhang/Packages
LEGION_ROOT_DIR=$PACKAGE_ROOT_DIR/legion
HDF5_ROOT_DIR=$PACKAGE_ROOT_DIR/HDF5_for_Legion
export CPLUS_INCLUDE_PATH=$HDF5_ROOT_DIR/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$HDF5_ROOT_DIR/lib:$LD_LIBRARY_PATH

CWD=$PWD

mkdir -p $LEGION_ROOT_DIR/build && cd $LEGION_ROOT_DIR/build

rm -rf ./*

HDF5_ROOT=$HDF5_ROOT_DIR \
cmake \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT_DIR/LegionBasics/\
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_SHARED_LIBS=True \
    -DLegion_SPY=True \
    -DLegion_BOUNDS_CHECKS=True \
    -DLegion_PRIVILEGE_CHECKS=True \
    -DLegion_USE_HDF5=True \
    -DLegion_USE_OpenMP=True \
    -DLegion_USE_CUDA=True \
    -DLegion_CUDA_ARCH="60" \
    -DLegion_MAX_DIM=3 \
    -DLegion_NETWORKS="gasnetex" \
    -DLegion_EMBED_GASNet=True \
    -DGASNet_CONDUIT=ibv \
    -DLegion_EMBED_GASNet_GITREF="3903e0f417393c33f481f10eaa547f2306d8ed5d" \
    ..

cd $CWD
