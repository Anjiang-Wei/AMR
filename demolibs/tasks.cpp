
#include "tasks.h"


void registerAllTasks() {

}



void taskTopLevel(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;

    BaseGridConfig grid_config = {
        8,   // PATCH_SIZE
        4,   // NUM_PATCHES_X
        4,   // NUM_PATCHES_Y
        5,   // STENCIL_WIDTH
        1.0, // LX
        1.0, // LY
    };

    constexpr unsigned int NUM_REGISTERS = 3;

    RegionOfFields coords; // coordinates
    RegionOfFields c_vars; // conservative variables
    RegionOfFields p_vars; // primitive variables

    IndexSpace color_isp_int = getColorIndexSpaceInt(ctx, rt, grid_config);
    initializeBaseGrid2D(ctx, rt, grid_config, static_cast<uint_t>(CVARS_ID::SIZE)*NUM_REGISTERS, c_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, static_cast<uint_t>(PVARS_ID::SIZE)              , p_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, static_cast<uint_t>(COORD_ID::SIZE)              , coords);
}



void taskMeshGen(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    const PhysicalRegion& rgn_coords = rgns[0];
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    // TODO Need to know LX and LY for mesh generation

    const FieldAccessor<WRITE_DISCARD, Real, 2> x (rgn_coords, static_cast<int>(COORD_ID::X));
    const FieldAccessor<WRITE_DISCARD, Real, 2> y (rgn_coords, static_cast<int>(COORD_ID::Y));
    const FieldAccessor<WRITE_DISCARD, Real, 2> z (rgn_coords, static_cast<int>(COORD_ID::Z));

    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        x[ij] = 0.0;
        y[ij] = 0.0;
        z[ij] = 0.0;
    }

}



void taskSetInitialCondition(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    const PhysicalRegion& rgn_coords = rgns[0];
    const PhysicalRegion& rgn_pvars  = rgns[1];

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    const FieldAccessor<READ_ONLY, Real, 2> x (rgn_coords, static_cast<int>(COORD_ID::X));
    const FieldAccessor<READ_ONLY, Real, 2> y (rgn_coords, static_cast<int>(COORD_ID::Y));
    const FieldAccessor<READ_ONLY, Real, 2> z (rgn_coords, static_cast<int>(COORD_ID::Z));

    const FieldAccessor<WRITE_DISCARD, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
    const FieldAccessor<WRITE_DISCARD, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );

    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        rho[ij] = 1.0;
        u  [ij] = 0.0;
        v  [ij] = 0.0;
        T  [ij] = 1.0;
    }
}




void taskConvertPrimitiveToConservative(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    // TODO: Need to pass an integer for stage
    const int CVAR_STAGE = 1;
}



void taskConvertConservativeToPrimitive(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    // TODO: Need to pass an integer for stage
    const int CVAR_STAGE = 1;
}
