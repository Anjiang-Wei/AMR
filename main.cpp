#include <stdio.h>
#include "util.h"
#include "numerics.h"

constexpr unsigned long PATCH_SIZE    = 8;
constexpr unsigned long NUM_PATCHES_X = 8;
constexpr unsigned long NUM_PATCHES_Y = 8;
constexpr Real          t_final       = 10.0;
constexpr Real          viscosity     = 10.0;
constexpr Real          Prandtl       = 1.0;
constexpr Real          Rg            = 1.0;
constexpr Real          gamma         = 1.4;

// Fieldspace of primitive variables
enum class PVARS_ID {
    VEL_X, VEL_Y, DENSITY, TEMP,
    SIZE
};


// Fieldspace of conservative variables
enum class STAGE3_CVARS_ID {
   MASS_0, MMTX_0, MMTY_0, ENRG_0, 
   MASS_1, MMTX_1, MMTY_1, ENRG_1, 
   MASS_2, MMTX_2, MMTY_2, ENRG_2, 
   SIZE
};

// Fieldspace of buffer variables
enum class BVARS_ID {
    PRSES,
    SIZE
};


// Mesh
enum class COORD_ID {
    X, Y, Z, SIZE
};


enum TASK_ID {
    TOP_LEVEL, ELEM_OP
};

void initBaseGrid(Context& ctx, Runtime* rt, const int fs_size, LogicalRegion& region_of_fields, LogicalPartition& patches)
{
    Box2D grid_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X*PATCH_SIZE-1, NUM_PATCHES_Y*PATCH_SIZE-1));
    IndexSpace grid_isp = rt->create_index_space(ctx, grid_bounds);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fs);
    for (int i = 0; i < fs_size; i++) {
        const int fid = field_allocator.allocate_field(sizeof(Real), i);
        assert(fid == i);
    }
    region_of_fields = rt->create_logical_region(ctx, grid_isp, fs);

    Box2D color_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X, NUM_PATCHES_Y));
    IndexSpace color_isp = rt->create_index_space(ctx, color_bounds);
    std::map<DomainPoint, Domain> domain_map;
    for (auto ip = 0; ip < NUM_PATCHES_X; ip++) {
        const unsigned long i_lo = ip * PATCH_SIZE;
        const unsigned long i_hi = i_lo + PATCH_SIZE - 1;
        for (auto jp = 0; jp < NUM_PATCHES_Y; jp++) {
            const unsigned long j_lo = jp * PATCH_SIZE;
            const unsigned long j_hi = j_lo + PATCH_SIZE - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map.insert({p_coord, patch});
        }
    }
    IndexPartition idx_partition = rt->create_partition_by_domain(ctx, grid_isp, domain_map, color_isp);
    patches = rt->get_logical_partition(ctx, region_of_fields, idx_partition);
}


//template<typename Op>
//void elem_op_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    Op op_functor = task->args;
//}

void set_initial_condition_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    // Set primitive variables
    {
        const FieldAccessor<WRITE_DISCARD, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
        const FieldAccessor<WRITE_DISCARD, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
        const FieldAccessor<WRITE_DISCARD, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
        const FieldAccessor<WRITE_DISCARD, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
        for (PointInBox2D ij(isp_domain); ij(); ij++) {
            rho[*ij] = 1.0;
            u  [*ij] = 0.0;
            v  [*ij] = 0.0;
            T  [*ij] = 1.0;
        }
    }

    // Convert primitive variables to conservative variables
    {
        const FieldAccessor<READ_ONLY, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
        const FieldAccessor<READ_ONLY, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
        const FieldAccessor<READ_ONLY, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
        const FieldAccessor<READ_ONLY, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
        const FieldAccessor<WRITE_DISCARD, Real, 2> mass (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MASS_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> mmtx (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTX_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> mmty (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTY_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> enrg (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::ENRG_0));
        for (PointInBox2D ij(isp_domain); ij(); ij++) {
            mass[*ij] = rho[*ij];
            mmtx[*ij] =   u[*ij];
            mmty[*ij] =   v[*ij];
            enrg[*ij] =   T[*ij];
        }
    }
}


void calc_rhs_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    constexpr int OVARS_RGN_ID = 0;
    constexpr int CVARS_RGN_ID = 1;
    constexpr int PVARS_RGN_ID = 2;
    constexpr int BVARS_RGN_ID = 3;

    // TODO: for now only set zeros
    Real* args = reinterpret_cast<Real*>(task->args);
    const Real arg0 = args[0];

}



void ssp_rk3_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    // TODO: Launch calc_rhs_task 3 times but using different stages of CVARS
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;

    LogicalRegion    rgn_coord;
    LogicalRegion    rgn_cvars;
    LogicalRegion    rgn_pvars;
    LogicalRegion    rgn_bvars;
    LogicalPartition patches_coord;
    LogicalPartition patches_cvars;
    LogicalPartition patches_pvars;
    LogicalPartition patches_bvars;

    printf("Call \"initBaseGrid\" from \"top_level_task\" ... ");
    initBaseGrid(ctx, rt, static_cast<int>(STAGE3_CVARS_ID::SIZE), rgn_cvars, patches_cvars);
    initBaseGrid(ctx, rt, static_cast<int>(       PVARS_ID::SIZE), rgn_pvars, patches_pvars);
    initBaseGrid(ctx, rt, static_cast<int>(       BVARS_ID::SIZE), rgn_bvars, patches_bvars);
    initBaseGrid(ctx, rt, static_cast<int>(       COORD_ID::SIZE), rgn_coord, patches_coord);
    printf("Done!\n");

    printf("\n\nSet initial conditions ... ");
    // TODO: index launch "set_initial_condition_task"
    printf("Done!\n");

    printf("\n\nStart time integration:\n");
    printf("Simulation done!!!\n");


}




int main(int argc, char* argv[]) {

    printf("Hello World from the real main test!\n");

    Legion::Runtime::set_top_level_task_id(TASK_ID::TOP_LEVEL);

    {
        Legion::TaskVariantRegistrar registrar(TASK_ID::TOP_LEVEL, "top_level_task");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
    }

    return Legion::Runtime::start(argc, argv);

}



