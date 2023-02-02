
#include "tasks.h"


void registerAllTasks() {

}



/*!
 * Root task.
 */
void taskTopLevel(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) rgns; (void) task; (void) ctx; (void) rt;
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;
    (void) argc; (void) argv; 

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

    //IndexSpace color_isp_int = getColorIndexSpaceInt(ctx, rt, grid_config);
    initializeBaseGrid2D(ctx, rt, grid_config, CVARS_ID::SIZE*NUM_REGISTERS, c_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, PVARS_ID::SIZE              , p_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, COORD_ID::SIZE              , coords);
}



/*!
 * Generate 2D Cartesian mesh.
 *
 * Fields:
 *     [wo][0] COORD::X
 *     [wo][0] COORD::Y
 *     [wo][0] COORD::Z
 *
 * Args: ArgsMeshGen
 */
void taskMeshGen(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) ctx; (void) rt;

    const PhysicalRegion& rgn_coords = rgns[0];
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    auto args = reinterpret_cast<ArgsMeshGen*>(task->args);
    const Real    Lx       = args->Lx;
    const Real    Ly       = args->Ly;
    const Real    offset_x = args->offset_x;
    const Real    offset_y = args->offset_y;
    const uint_t  Nx       = args->Nx;
    const uint_t  Ny       = args->Ny;
    const Real    dx       = Lx / static_cast<Real>(Nx);
    const Real    dy       = Ly / static_cast<Real>(Ny);
    const Point2D origin   = args->origin;

    const FieldAccessor<WRITE_DISCARD, Real, 2> x (rgn_coords, COORD_ID::X);
    const FieldAccessor<WRITE_DISCARD, Real, 2> y (rgn_coords, COORD_ID::Y);

    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        x[ij] = (ij[0] - origin[0] + 0.5) * dx - offset_x;
        y[ij] = (ij[1] - origin[1] + 0.5) * dy - offset_y;
    }

}



/*!
 * Set initial condition to primitive variables.
 *
 * Fields:
 *    [ro][0] COORD::X
 *    [ro][0] COORD::Y
 *    [ro][0] COORD::Z
 *    [wo][1] PVARS::DENSITY
 *    [wo][1] PVARS::VEL_X
 *    [wo][1] PVARS::VEL_Y
 *    [wo][1] PVARS::TEMP
 */
void taskSetInitialCondition(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) task; (void) ctx;

    const PhysicalRegion& rgn_coords = rgns[0];
    const PhysicalRegion& rgn_pvars  = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> x (rgn_coords, COORD_ID::X);
    const FieldAccessor<READ_ONLY, Real, 2> y (rgn_coords, COORD_ID::Y);

    const FieldAccessor<WRITE_DISCARD, Real, 2> rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> T  (rgn_pvars, PVARS_ID::TEMP   );

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        rho[ij] = 1.0;
        u  [ij] = 0.0;
        v  [ij] = 0.0;
        T  [ij] = 1.0;
    }
}



/*!
 * Convert primitive variables to conservative variables.
 *
 * Fields:
 * [ro][0] PVARS::DENSITY
 * [ro][0] PVARS::VEL_X
 * [ro][0] PVARS::VEL_Y
 * [ro][0] PVARS::TEMP
 * [wo][1] CVARS::DENSITY
 * [wo][1] CVARS::VEL_X
 * [wo][1] CVARS::VEL_Y
 * [wo][1] CVARS::TEMP
 *
 * Args: ArgsConvertPrimitiveToConservative
 */
void taskConvertPrimitiveToConservative(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) ctx;

    const PhysicalRegion& rgn_pvars = rgns[0];
    const PhysicalRegion& rgn_cvars = rgns[1];

    auto args = reinterpret_cast<ArgsConvertPrimitiveToConservative*>(task->args);
    const int STAGE_ID = args->stage_id;
    //const EquationOfState

    const FieldAccessor<READ_ONLY, Real, 2> p_rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2> p_u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2> p_v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2> p_T  (rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rho (rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhou(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhov(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_Etot(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::ENRG);

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        c_rho [ij] = p_rho[ij];
        c_rhou[ij] = p_rho[ij] * p_u[ij];
        c_rhov[ij] = p_rho[ij] * p_v[ij];
        //c_Etot[ij] = p_rho[ij] * ();
    }

}



/*!
 * Convert conservative variables to primitive variables.
 *
 * Fields:
 * [ro][0] CVARS::DENSITY
 * [ro][0] CVARS::VEL_X
 * [ro][0] CVARS::VEL_Y
 * [ro][0] CVARS::TEMP
 * [wo][1] PVARS::DENSITY
 * [wo][1] PVARS::VEL_X
 * [wo][1] PVARS::VEL_Y
 * [wo][1] PVARS::TEMP
 *
 * Args: ArgsConvertConservativeToPrimitive
 */
void taskConvertConservativeToPrimitive(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) task; (void) rgns; (void) ctx; (void) rt; // TODO: delete this line

    // TODO: Need to pass an integer for stage
    const int CVAR_STAGE = 1;
    (void)  CVAR_STAGE;
}
