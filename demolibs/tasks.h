
#ifndef _TASKS_H
#define _TASKS_H

#include "legion_hook.h"
#include "util.h"
#include <vector>

enum class TASK_ID : int {
    TOP_LEVEL, MESH_GEN, SET_INIT_COND,
    SIZE
};

enum class CVARS_ID : int {
    MASS, MMTX, MMTY, ENRG,
    SIZE 
};

enum class PVARS_ID {
    VEL_X, VEL_Y, DENSITY, TEMP,
    SIZE
};

enum class COORD_ID {
    X, Y, Z, SIZE
};

void registerAllTasks();

void taskTopLevel                      (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskMeshGen                       (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskSetInitialCondition           (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertPrimitiveToConservative(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertConservativeToPrimitive(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#endif
