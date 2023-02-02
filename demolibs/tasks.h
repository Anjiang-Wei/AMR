
#ifndef _TASKS_H
#define _TASKS_H

#include "legion_hook.h"
#include "util.h"
#include "num_kernels.h"
#include "physical_models.h"
#include <vector>

struct CVARS_ID {
    enum {
        MASS, MMTX, MMTY, ENRG,
        SIZE 
    };
};

struct  PVARS_ID {
    enum {
        VEL_X, VEL_Y, DENSITY, TEMP,
        SIZE
    };
};

struct  COORD_ID {
    enum {X, Y, SIZE};
};


struct TASK_ID {
    enum {
        TOP_LEVEL, MESH_GEN, SET_INIT_COND,
        SIZE
    };
};


void registerAllTasks();
void taskTopLevel                      (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskMeshGen                       (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskSetInitialCondition           (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertPrimitiveToConservative(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertConservativeToPrimitive(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

struct ArgsMeshGen{
    Real Lx;
    Real Ly;
    Real offset_x = 0.0;
    Real offset_y = 0.0;
    uint_t Nx;
    uint_t Ny;
    Point2D origin;
};

struct ArgsConvertPrimitiveToConservative{ int stage_id; PerfectGasModel eos; };
struct ArgsConvertConservativeToPrimitive{ int stage_id; PerfectGasModel eos; };

#endif
