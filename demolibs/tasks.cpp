
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
    (void) ctx;

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

    const PhysicalRegion& rgn_coords = rgns[0];
    const FieldAccessor<WRITE_DISCARD, Real, 2> x (rgn_coords, COORD_ID::X);
    const FieldAccessor<WRITE_DISCARD, Real, 2> y (rgn_coords, COORD_ID::Y);

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
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

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
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

    auto args = reinterpret_cast<ArgsConvertPrimitiveToConservative*>(task->args);
    const int STAGE_ID = args->stage_id;
    auto& eos = args->eos;

    const PhysicalRegion& rgn_pvars = rgns[0];
    const PhysicalRegion& rgn_cvars = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> p_rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2> p_u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2> p_v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2> p_T  (rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rho (rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhou(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhov(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_Etot(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::ENRG);

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        c_rho [ij] = p_rho[ij];
        c_rhou[ij] = p_rho[ij] * p_u[ij];
        c_rhov[ij] = p_rho[ij] * p_v[ij];
        c_Etot[ij] = p_rho[ij] * (eos.calcInternalEnergyEVT(p_rho[ij], p_T[ij]) + 0.5 * (p_u[ij]*p_u[ij] + p_v[ij]*p_v[ij]));
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
    (void) ctx;

    auto args = reinterpret_cast<ArgsConvertConservativeToPrimitive*>(task->args);
    const int STAGE_ID = args->stage_id;
    auto& eos = args->eos;

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> c_rho (rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MASS);
    const FieldAccessor<READ_ONLY, Real, 2> c_rhou(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTX);
    const FieldAccessor<READ_ONLY, Real, 2> c_rhov(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::MMTY);
    const FieldAccessor<READ_ONLY, Real, 2> c_Etot(rgn_cvars, STAGE_ID * CVARS_ID::SIZE + CVARS_ID::ENRG);

    const FieldAccessor<WRITE_DISCARD, Real, 2> p_rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> p_u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> p_v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<WRITE_DISCARD, Real, 2> p_T  (rgn_pvars, PVARS_ID::TEMP   );

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        const Real rho_inv = 1.0 / c_rho[ij];
        const Real u       = c_rhou[ij] * rho_inv;
        const Real v       = c_rhov[ij] * rho_inv;
        const Real e_int   = c_Etot[ij] * rho_inv - 0.5 * (u*u + v*v);
        p_rho [ij] = c_rho[ij];
        p_u   [ij] = u;
        p_v   [ij] = v;
        p_T   [ij] = eos.calcTemperatureEVT(e_int, c_rho[ij]);
    }
}




/*!
 * Calculate the right-hand sides of evolution equations
 *
 * Fields:
 * [rw][0] CVARS::DENSITY
 * [rw][0] CVARS::VEL_X
 * [rw][0] CVARS::VEL_Y
 * [rw][0] CVARS::TEMP
 * [rw][1] PVARS::DENSITY
 * [rw][1] PVARS::VEL_X
 * [rw][1] PVARS::VEL_Y
 * [rw][1] PVARS::TEMP
 *
 * Args: ArgsCalcRHS
 */
void taskCalcRHS(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) ctx;

    auto args = reinterpret_cast<ArgsCalcRHS*>(task->args);
    const int STAGE_ID_NOW = args->stage_id_now;
    const int STAGE_ID_DDT = args->stage_id_ddt;
    auto               eos = args->eos;
    const Real      dx_inv = 1.0 / args->dx;
    const Real      dy_inv = 1.0 / args->dy;

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];


    // TODO: Inline launch taskConvertConservativeToPrimitive
    ArgsConvertPrimitiveToConservative args_convert_primitive_to_conservative {STAGE_ID_NOW, eos};
    (void) args_convert_primitive_to_conservative; // TODO: delete this line

    //const FieldAccessor<READ_ONLY, Real, 2> c_mass(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MASS);
    //const FieldAccessor<READ_ONLY, Real, 2> c_mmtx(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MMTX);
    //const FieldAccessor<READ_ONLY, Real, 2> c_mmty(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MMTY);
    //const FieldAccessor<READ_ONLY, Real, 2> c_enrg(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::ENRG);

    const FieldAccessor<READ_ONLY, Real, 2> rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2> u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2> v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2> T  (rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mass(rgn_cvars, STAGE_ID_DDT * CVARS_ID::SIZE + CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmtx(rgn_cvars, STAGE_ID_DDT * CVARS_ID::SIZE + CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmty(rgn_cvars, STAGE_ID_DDT * CVARS_ID::SIZE + CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_enrg(rgn_cvars, STAGE_ID_DDT * CVARS_ID::SIZE + CVARS_ID::ENRG);


    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij   = *pir;
        Point2D ije4 = ij + Point2D(4,0);
        Point2D ije3 = ij + Point2D(3,0);
        Point2D ije2 = ij + Point2D(2,0);
        Point2D ije1 = ij + Point2D(1,0);
        Point2D ijw1 = ij - Point2D(1,0);
        Point2D ijw2 = ij - Point2D(2,0);
        Point2D ijw3 = ij - Point2D(3,0);
        Point2D ijw4 = ij - Point2D(4,0);
        Point2D ijn4 = ij + Point2D(0,4);
        Point2D ijn3 = ij + Point2D(0,3);
        Point2D ijn2 = ij + Point2D(0,2);
        Point2D ijn1 = ij + Point2D(0,1);
        Point2D ijs1 = ij - Point2D(0,1);
        Point2D ijs2 = ij - Point2D(0,2);
        Point2D ijs3 = ij - Point2D(0,3);
        Point2D ijs4 = ij - Point2D(0,4);

        /*** Step 1 assemble fluxes in x-staggered ***/
        {
            Real flx_mass_w2, flx_mass_w1, flx_mass_e1, flx_mass_e2;
            Real flx_mmtx_w2, flx_mmtx_w1, flx_mmtx_e1, flx_mmtx_e2;
            Real flx_mmty_w2, flx_mmty_w1, flx_mmty_e1, flx_mmty_e2;
            Real flx_enrg_w2, flx_enrg_w1, flx_enrg_e1, flx_enrg_e2;

            Real rho_stag, u_stag, v_stag, T_stag, p_stag;
            u_stag = ei04Stag(u[ij  ], u[ije1], u[ije2], u[ije3]);
            v_stag = ei04Stag(v[ij  ], v[ije1], v[ije2], v[ije3]);
            T_stag = ei04Stag(T[ij  ], T[ije1], T[ije2], T[ije3]);

            flx_mass_e2 = -rho_stag * u_stag;

            Real rho_e2 = ei04Stag(rho[ij  ], rho[ije1], rho[ije2], rho[ije3]);
            Real rho_e1 = ei04Stag(rho[ijw1], rho[ij  ], rho[ije1], rho[ije2]);
            Real rho_w1 = ei04Stag(rho[ijw2], rho[ijw1], rho[ij  ], rho[ije1]);
            Real rho_w2 = ei04Stag(rho[ijw3], rho[ijw2], rho[ijw1], rho[ij  ]);

            Real u_e2 = ei04Stag(u[ij  ], u[ije1], u[ije2], u[ije3]);
            Real u_e1 = ei04Stag(u[ijw1], u[ij  ], u[ije1], u[ije2]);
            Real u_w1 = ei04Stag(u[ijw2], u[ijw1], u[ij  ], u[ije1]);
            Real u_w2 = ei04Stag(u[ijw3], u[ijw2], u[ijw1], u[ij  ]);

            Real v_e2 = ei04Stag(v[ij  ], v[ije1], v[ije2], v[ije3]);
            Real v_e1 = ei04Stag(v[ijw1], v[ij  ], v[ije1], v[ije2]);
            Real v_w1 = ei04Stag(v[ijw2], v[ijw1], v[ij  ], v[ije1]);
            Real v_w2 = ei04Stag(v[ijw3], v[ijw2], v[ijw1], v[ij  ]);

            Real T_e2 = ei04Stag(T[ij  ], T[ije1], T[ije2], T[ije3]);
            Real T_e1 = ei04Stag(T[ijw1], T[ij  ], T[ije1], T[ije2]);
            Real T_w1 = ei04Stag(T[ijw2], T[ijw1], T[ij  ], T[ije1]);
            Real T_w2 = ei04Stag(T[ijw3], T[ijw2], T[ijw1], T[ij  ]);
        }

    }
}





