
#ifndef _TASKS_H
#define _TASKS_H

#include "legion_hook.h"
#include "util.h"
#include "num_kernels.h"
#include "physical_models.h"
#include "hdf5_hook.h"
#include <vector>

//constexpr int STENCIL_SIZE = 5;
//constexpr unsigned int NUM_REGISTERS = 3; // number of registers for timestepping

struct CVARS_ID {
    enum {
        MASS, MMTX, MMTY, ENRG,
        SIZE 
    };
};

struct  PVARS_ID {
    enum {
        DENSITY, VEL_X, VEL_Y, TEMP,
        SIZE
    };
};

struct  COORD_ID {
    enum {X, Y, SIZE};
};


struct TASK_ID {
    enum {
        TOP_LEVEL,
        MESH_GEN,
        SET_INIT_COND,
        PVARS_TO_CVARS,
        CVARS_TO_PVARS,
        CALC_RHS,
        SSPRK3_LINCOMB_1,
        SSPRK3_LINCOMB_2,
        COPY_PVARS,
        SIZE
    };
};

struct ArgsMeshGen {
    Real Lx;
    Real Ly;
    Real offset_x = 0.0;
    Real offset_y = 0.0;
    uint_t Nx;
    uint_t Ny;
    Point2D origin;
};

struct ArgsConvertPrimitiveToConservative{ Real R_gas; Real gamma; };
struct ArgsConvertConservativeToPrimitive{ Real R_gas; Real gamma; };

struct ArgsCalcRHS {
    Real dx;
    Real dy;
    Real dt;
    int stage_id_now;
    int stage_id_ddt;
    int stencil_size;
    Real R_gas;
    Real gamma;
    Real mu_ref;
    Real  T_ref;
    Real visc_exp;
    Real Pr;
};


struct ArgsSolve {
    Real R_gas;
    Real gamma;
    Real mu_ref;
    Real  T_ref;
    Real visc_exp;
    Real Pr;
    Real dt;
    Real dx;
    Real dy;
    uint_t num_iter;
    uint_t output_freq;
    int stencil_size;
};

struct ArgsCopyPrimVars { int offset_x; int offset_y; };


void registerAllTasks();
void taskTopLevel                      (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskMeshGen                       (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskSetInitialCondition           (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertPrimitiveToConservative(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskConvertConservativeToPrimitive(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskCalcRHS                       (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskSSPRK3LinearCombination1      (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskSSPRK3LinearCombination2      (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void taskCopyPrimVars                  (const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
void fillGhosts(const IndexSpace&, const IndexSpace&, RegionOfFields&, std::vector<unsigned int>, Context, Runtime*);
void launchSSPRK3(IndexSpace&, IndexSpace&, IndexSpace&, RegionOfFields&, RegionOfFields&, RegionOfFields&, RegionOfFields&, const ArgsSolve&, const BaseGridConfig&,  Context, Runtime*); 
void fillGhostsNew(RegionOfFields&, const BaseGridConfig&, const IndexSpace&, const IndexSpace&, Context, Runtime*);


#endif
