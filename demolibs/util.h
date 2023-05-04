#ifndef _UTIL_H
#define _UTIL_H

#include <cstdio>
#include "legion_hook.h"

typedef double Real;
typedef unsigned long long uint_t;
typedef long long int int_t;

struct BaseGridConfig {
    const uint_t PATCH_SIZE;
    const uint_t NUM_PATCHES_X;
    const uint_t NUM_PATCHES_Y;
    const uint_t STENCIL_WIDTH;
    const Real LX;
    const Real LY;
};

struct RegionOfFields {
    LogicalRegion    region;
    LogicalPartition patches_int;
    LogicalPartition patches_ext;
    LogicalPartition patches_ghost_x_lo;
    LogicalPartition patches_ghost_y_lo;
    LogicalPartition patches_ghost_mirror_x_lo;
    LogicalPartition patches_ghost_mirror_y_lo;
    LogicalPartition patches_ghost_x_hi;
    LogicalPartition patches_ghost_y_hi;
    LogicalPartition patches_ghost_mirror_x_hi;
    LogicalPartition patches_ghost_mirror_y_hi;
};

IndexSpace getColorIndexSpaceInt(Context&, Runtime*, const BaseGridConfig&);
IndexSpace getColorIndexSpaceExt(Context&, Runtime*, const BaseGridConfig&);

void initializeBaseGrid2D(Context&, Runtime*, const BaseGridConfig, const uint_t, RegionOfFields&);

#endif
