
#include "tasks.h"
#include <cmath>


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


    RegionOfFields coords; // coordinates
    RegionOfFields c0_vars; // conservative variables
    RegionOfFields c1_vars; // conservative variables
    RegionOfFields c2_vars; // conservative variables
    RegionOfFields p_vars; // primitive variables

    //IndexSpace color_isp_int = getColorIndexSpaceInt(ctx, rt, grid_config);
    initializeBaseGrid2D(ctx, rt, grid_config, CVARS_ID::SIZE, c0_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, CVARS_ID::SIZE, c1_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, CVARS_ID::SIZE, c2_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, PVARS_ID::SIZE, p_vars);
    initializeBaseGrid2D(ctx, rt, grid_config, COORD_ID::SIZE, coords);
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
    const Real R_gas    = args->R_gas;
    const Real gamm_inv = 1.0 / (args->gamma - 1.0);

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
        const Real e_int = R_gas * gamm_inv * p_T[ij];
        c_rho [ij] = p_rho[ij];
        c_rhou[ij] = p_rho[ij] * p_u[ij];
        c_rhov[ij] = p_rho[ij] * p_v[ij];
        c_Etot[ij] = p_rho[ij] * (e_int + 0.5 * (p_u[ij]*p_u[ij] + p_v[ij]*p_v[ij]));
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
    const Real R_gas  = args->R_gas;
    const Real gamm   = args->gamma - 1.0;
    const Real Rg_inv = 1.0 / R_gas;

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
        p_T   [ij] = gamm * e_int * Rg_inv;
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
    const Real          Rg = args->R_gas;
    const Real         gam = args->gamma;
    const Real      mu_ref = args->mu_ref;
    const Real       T_ref = args->T_ref;
    const Real    visc_exp = args->visc_exp;
    const Real          Pr = args->Pr;
    const Real      dx_inv = 1.0 / args->dx;
    const Real      dy_inv = 1.0 / args->dy;

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];


    // TODO: Inline launch taskConvertConservativeToPrimitive
    //ArgsConvertConservativeToPrimitive args_convert_conservative_to_primitive {STAGE_ID_NOW, eos};
    //(void) args_convert_conservative_to_primitive; // TODO: delete this line

    //const FieldAccessor<READ_ONLY, Real, 2> c_mass(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MASS);
    //const FieldAccessor<READ_ONLY, Real, 2> c_mmtx(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MMTX);
    //const FieldAccessor<READ_ONLY, Real, 2> c_mmty(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::MMTY);
    //const FieldAccessor<READ_ONLY, Real, 2> c_enrg(rgn_cvars, STAGE_ID_NOW * CVARS_ID::SIZE + CVARS_ID::ENRG);

    const FieldAccessor<READ_ONLY, Real, 2> rho_coll(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2>   u_coll(rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2>   v_coll(rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2>   T_coll(rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mass(rgn_cvars, CVARS_ID::SIZE + CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmtx(rgn_cvars, CVARS_ID::SIZE + CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmty(rgn_cvars, CVARS_ID::SIZE + CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_enrg(rgn_cvars, CVARS_ID::SIZE + CVARS_ID::ENRG);

    constexpr int EDGE_STENCIL_SIZE = STENCIL_SIZE - 1;
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij   = *pir;

        Real flx_mass[EDGE_STENCIL_SIZE], flx_mmtx[EDGE_STENCIL_SIZE], flx_mmty[EDGE_STENCIL_SIZE], flx_enrg[EDGE_STENCIL_SIZE];
        Real rho, u, v, w, T, p, h, dudx, dudy, dvdx, dvdy, gradT;
        Real du_coll[EDGE_STENCIL_SIZE], dv_coll[EDGE_STENCIL_SIZE];

        /*** Step 1 assemble fluxes in x-staggered ***/
        for (int idx_edge = 0; idx_edge < EDGE_STENCIL_SIZE; idx_edge++) {
            Point2D ijw2 = ij + Point2D(idx_edge-3, 0); 
            Point2D ijw1 = ij + Point2D(idx_edge-2, 0); 
            Point2D ije1 = ij + Point2D(idx_edge-1, 0); 
            Point2D ije2 = ij; 

            // Load variables at edges
            const Real p_coll_ijw2 = rho_coll[ijw2] * Rg * T_coll[ijw2];
            const Real p_coll_ijw1 = rho_coll[ijw1] * Rg * T_coll[ijw1];
            const Real p_coll_ije1 = rho_coll[ije1] * Rg * T_coll[ije1];
            const Real p_coll_ije2 = rho_coll[ije2] * Rg * T_coll[ije2];

            u = ei04Stag(u_coll[ijw2], u_coll[ijw1], u_coll[ije1], u_coll[ije2]);
            v = ei04Stag(v_coll[ijw2], v_coll[ijw1], v_coll[ije1], v_coll[ije2]);
            T = ei04Stag(T_coll[ijw2], T_coll[ijw1], T_coll[ije1], T_coll[ije2]);
            p = ei04Stag(p_coll_ijw2 , p_coll_ijw1 , p_coll_ije1 , p_coll_ije2 );

            rho = p / (Rg * T); 
            h = Rg * gam / (gam - 1.0) * T + 0.5 * (u*u + v*v);
            dudx  = ed04Stag(u_coll[ijw2], u_coll[ijw1], u_coll[ije1], u_coll[ije2], dx_inv);
            dvdx  = ed04Stag(v_coll[ijw2], v_coll[ijw1], v_coll[ije1], v_coll[ije2], dx_inv);
            gradT = ed04Stag(T_coll[ijw2], T_coll[ijw1], T_coll[ije1], T_coll[ije2], dx_inv);
            for (int idx_coll = 0; idx_coll < EDGE_STENCIL_SIZE; idx_coll++) {
                Point2D ij0  = ijw2 + Point2D(idx_coll, 0);
                Point2D ijn2 = ij0  + Point2D(0,  2);
                Point2D ijn1 = ij0  + Point2D(0,  1);
                Point2D ijs1 = ij0  + Point2D(0, -1);
                Point2D ijs2 = ij0  + Point2D(0, -2);
                du_coll[idx_coll] = ed04Coll(u_coll[ijs2], u_coll[ijs1], u_coll[ijn1], u_coll[ijn2], dy_inv);
                dv_coll[idx_coll] = ed04Coll(v_coll[ijs2], v_coll[ijs1], v_coll[ijn1], v_coll[ijn2], dy_inv);
            }
            dudy = ei04Stag(du_coll[0], du_coll[1], du_coll[2], du_coll[3]);
            dvdy = ei04Stag(dv_coll[0], dv_coll[1], dv_coll[2], dv_coll[3]);

            const Real mu  = mu_ref * pow(T/T_ref, visc_exp);
            const Real kap = Rg * gam / (gam - 1.0) * mu / Pr;
            const Real Skk = dudx + dvdy;
            const Real str11 = 2.0 * mu * dudx - (2./3.) * mu * Skk;
            const Real str21 = mu * (dudy + dvdx);

            // Assemble fluxes
            flx_mass[idx_edge] = -rho * u;
            flx_mmtx[idx_edge] = -rho * u * u - p + str11;
            flx_mmty[idx_edge] = -rho * v * u     + str21;
            flx_enrg[idx_edge] = -rho * h * u + u * str11 + v * str21;
        }
        const Real ddt_mass_x = ed04Stag(flx_mass[0], flx_mass[1], flx_mass[2], flx_mass[3], dx_inv);
        const Real ddt_mmtx_x = ed04Stag(flx_mmtx[0], flx_mmtx[1], flx_mmtx[2], flx_mmtx[3], dx_inv);
        const Real ddt_mmty_x = ed04Stag(flx_mmty[0], flx_mmty[1], flx_mmty[2], flx_mmty[3], dx_inv);
        const Real ddt_enrg_x = ed04Stag(flx_enrg[0], flx_enrg[1], flx_enrg[2], flx_enrg[3], dx_inv);

        /*** Step 2 assemble fluxes in y-staggered ***/
        for (int idx_edge = 0; idx_edge < EDGE_STENCIL_SIZE; idx_edge++) {
            Point2D ijs2 = ij + Point2D(0, idx_edge-3); 
            Point2D ijs1 = ij + Point2D(0, idx_edge-2); 
            Point2D ijn1 = ij + Point2D(0, idx_edge-1); 
            Point2D ijn2 = ij; 

            // Load variables at edges
            const Real p_coll_ijs2 = rho_coll[ijs2] * Rg * T_coll[ijs2];
            const Real p_coll_ijs1 = rho_coll[ijs1] * Rg * T_coll[ijs1];
            const Real p_coll_ijn1 = rho_coll[ijn1] * Rg * T_coll[ijn1];
            const Real p_coll_ijn2 = rho_coll[ijn2] * Rg * T_coll[ijn2];

            u = ei04Stag(u_coll[ijs2], u_coll[ijs1], u_coll[ijn1], u_coll[ijn2]);
            v = ei04Stag(v_coll[ijs2], v_coll[ijs1], v_coll[ijn1], v_coll[ijn2]);
            T = ei04Stag(T_coll[ijs2], T_coll[ijs1], T_coll[ijn1], T_coll[ijn2]);
            p = ei04Stag(p_coll_ijs2 , p_coll_ijs1 , p_coll_ijn1 , p_coll_ijn2 );
            rho = p / (Rg * T); 
            h = Rg * gam / (gam - 1.0) * T + 0.5 * (u*u + v*v);
            dudy  = ed04Stag(u_coll[ijs2], u_coll[ijs1], u_coll[ijn1], u_coll[ijn2], dy_inv);
            dvdy  = ed04Stag(v_coll[ijs2], v_coll[ijs1], v_coll[ijn1], v_coll[ijn2], dy_inv);
            gradT = ed04Stag(T_coll[ijs2], T_coll[ijs1], T_coll[ijn1], T_coll[ijn2], dy_inv);
            for (int idx_coll = 0; idx_coll < EDGE_STENCIL_SIZE; idx_coll++) {
                Point2D ij0  = ijs2 + Point2D(0, idx_coll);
                Point2D ije2 = ij0  + Point2D( 2, 0);
                Point2D ije1 = ij0  + Point2D( 1, 0);
                Point2D ijw1 = ij0  + Point2D(-1, 0);
                Point2D ijw2 = ij0  + Point2D(-2, 0);
                du_coll[idx_coll] = ed04Coll(u_coll[ijw2], u_coll[ijw1], u_coll[ije1], u_coll[ije2], dx_inv);
                dv_coll[idx_coll] = ed04Coll(v_coll[ijw2], v_coll[ijw1], v_coll[ije1], v_coll[ije2], dx_inv);
            }
            dudx = ei04Stag(du_coll[0], du_coll[1], du_coll[2], du_coll[3]);
            dvdx = ei04Stag(dv_coll[0], dv_coll[1], dv_coll[2], dv_coll[3]);

            const Real mu  = mu_ref * pow(T/T_ref, visc_exp);
            const Real kap = Rg * gam / (gam - 1.0) * mu / Pr;
            const Real Skk = dudx + dvdy;
            const Real str12 = mu * (dudy + dvdx);
            const Real str22 = 2.0 * mu * dvdy - (2./3.) * mu * Skk;

            // Assemble fluxes
            flx_mass[idx_edge] = -rho * v;
            flx_mmtx[idx_edge] = -rho * u * v     + str12;
            flx_mmty[idx_edge] = -rho * v * v -p  + str22;
            flx_enrg[idx_edge] = -rho * h * v + u * str12 + v * str22;
        }
        ddt_mass[ij] = ddt_mass_x + ed04Stag(flx_mass[0], flx_mass[1], flx_mass[2], flx_mass[3], dy_inv);
        ddt_mmtx[ij] = ddt_mmtx_x + ed04Stag(flx_mmtx[0], flx_mmtx[1], flx_mmtx[2], flx_mmtx[3], dy_inv);
        ddt_mmty[ij] = ddt_mmty_x + ed04Stag(flx_mmty[0], flx_mmty[1], flx_mmty[2], flx_mmty[3], dy_inv);
        ddt_enrg[ij] = ddt_enrg_x + ed04Stag(flx_enrg[0], flx_enrg[1], flx_enrg[2], flx_enrg[3], dy_inv);
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
void taskSSPRK3(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    const PhysicalRegion& rgn_cvars_stage0 = rgns[0];
    const PhysicalRegion& rgn_cvars_stage1 = rgns[1];
    const PhysicalRegion& rgn_cvars_stage2 = rgns[2];
    const PhysicalRegion& rgn_pvars        = rgns[3];

}
