
#include "tasks.h"
#include <cmath>
#include <cstdio>


void registerAllTasks() {
    Legion::Runtime::set_top_level_task_id(static_cast<int>(TASK_ID::TOP_LEVEL));

    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::TOP_LEVEL), "top_level_task");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskTopLevel>(registrar, "top_level_task");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::MESH_GEN), "mesh generation");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskMeshGen>(registrar, "mesh generation");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::SET_INIT_COND), "set initial condition");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskSetInitialCondition>(registrar, "set initial condition");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::PVARS_TO_CVARS), "primitive to conservative");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskConvertPrimitiveToConservative>(registrar, "primitive to conservative");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::CVARS_TO_PVARS), "conservative to primitive");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskConvertConservativeToPrimitive>(registrar, "conservative to primitive");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::CALC_RHS), "calculate rhs");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskCalcRHS>(registrar, "calculate rhs");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::SSPRK3_LINCOMB_1), "SSPRK3 linear combination 1");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskSSPRK3LinearCombination1>(registrar, "SSPRK3 linear combination 1");
    }
    {
        Legion::TaskVariantRegistrar registrar(static_cast<int>(TASK_ID::SSPRK3_LINCOMB_2), "SSPRK3 linear combination 2");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<taskSSPRK3LinearCombination2>(registrar, "SSPRK3 linear combination 2");
    }
}



/*!
 * Root task.
 */
void taskTopLevel(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) rgns; (void) task; (void) ctx; (void) rt;
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;
    (void) argc; (void) argv; 

    /********************************************************************************/
    BaseGridConfig grid_config = {
        8,   // PATCH_SIZE
        4,   // NUM_PATCHES_X
        4,   // NUM_PATCHES_Y
        5,   // STENCIL_WIDTH
        1.0, // LX
        1.0, // LY
    };

    ArgsSolve args_solve;
    args_solve.R_gas        = 1.0;
    args_solve.gamma        = 1.4;
    args_solve.dt           = 1e-3;
    args_solve.dx           = 1.0 / (grid_config.PATCH_SIZE * grid_config.NUM_PATCHES_X);
    args_solve.dy           = 1.0 / (grid_config.PATCH_SIZE * grid_config.NUM_PATCHES_Y);
    args_solve.stencil_size = grid_config.STENCIL_WIDTH;
    args_solve.num_iter     = 1;
    /********************************************************************************/


    RegionOfFields coords;  // coordinates
    RegionOfFields c0_vars; // conservative variables
    RegionOfFields c1_vars; // conservative variables
    RegionOfFields c2_vars; // conservative variables
    RegionOfFields p_vars;  // primitive variables

    const std::vector<unsigned int> field_id_coords {0, 1};
    const std::vector<unsigned int> field_id_c_vars {0, 1, 2, 3};
    const std::vector<unsigned int> field_id_p_vars {0, 1, 2, 3};

    Box2D color_bounds_int     = Box2D(Point2D(0,0), Point2D(grid_config.NUM_PATCHES_X-1, grid_config.NUM_PATCHES_Y-1));
    Box2D color_bounds_ghost_x = Box2D(Point2D(0,0), Point2D(0, grid_config.NUM_PATCHES_Y-1));
    Box2D color_bounds_ghost_y = Box2D(Point2D(0,0), Point2D(grid_config.NUM_PATCHES_X-1, 0));
    IndexSpace color_isp_int     = rt->create_index_space(ctx, color_bounds_int);
    IndexSpace color_isp_ghost_x = rt->create_index_space(ctx, color_bounds_ghost_x);
    IndexSpace color_isp_ghost_y = rt->create_index_space(ctx, color_bounds_ghost_y);
    initializeBaseGrid2DNew(ctx, rt, grid_config, CVARS_ID::SIZE, c0_vars);
    initializeBaseGrid2DNew(ctx, rt, grid_config, CVARS_ID::SIZE, c1_vars);
    initializeBaseGrid2DNew(ctx, rt, grid_config, CVARS_ID::SIZE, c2_vars);
    initializeBaseGrid2DNew(ctx, rt, grid_config, PVARS_ID::SIZE, p_vars);
    initializeBaseGrid2DNew(ctx, rt, grid_config, COORD_ID::SIZE, coords);

    // Initialize fields
    rt->fill_fields(ctx, c0_vars.region, c0_vars.region, std::set<uint>(field_id_c_vars.begin(), field_id_c_vars.end()), 0.0);
    rt->fill_fields(ctx, c1_vars.region, c1_vars.region, std::set<uint>(field_id_c_vars.begin(), field_id_c_vars.end()), 0.0);
    rt->fill_fields(ctx, c2_vars.region, c2_vars.region, std::set<uint>(field_id_c_vars.begin(), field_id_c_vars.end()), 0.0);
    rt->fill_fields(ctx,  p_vars.region,  p_vars.region, std::set<uint>(field_id_p_vars.begin(), field_id_p_vars.end()), 0.0);
    rt->fill_fields(ctx,  coords.region,  coords.region, std::set<uint>(field_id_coords.begin(), field_id_coords.end()), 0.0);

    // Mesh generation
    { // Launch taskMeshGen
        printf("Generate computational mesh ... ");
        ArgsMeshGen args_mesh_gen;
        args_mesh_gen.Lx       = grid_config.LX;
        args_mesh_gen.Ly       = grid_config.LY;
        args_mesh_gen.offset_x = 0.0;
        args_mesh_gen.offset_y = 0.0;
        args_mesh_gen.Nx       = grid_config.PATCH_SIZE * grid_config.NUM_PATCHES_X;
        args_mesh_gen.Ny       = grid_config.PATCH_SIZE * grid_config.NUM_PATCHES_Y;
        args_mesh_gen.origin   = Point2D(0, 0);
        IndexLauncher launcher (
                TASK_ID::MESH_GEN,
                color_isp_int,
                TaskArgument(&args_mesh_gen, sizeof(ArgsMeshGen)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(coords.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, coords.region));
        launcher.region_requirements[0].add_fields(field_id_coords);
        rt->execute_index_space(ctx, launcher);
        printf("Done!\n");
    }

    // Set initial condition
    { // Launch taskSetInitialCondition
        printf("Set initial conditions ... ");
        IndexLauncher launcher (
                TASK_ID::SET_INIT_COND,
                color_isp_int,
                TaskArgument(NULL, 0),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(coords.patches_int, 0, READ_ONLY,     EXCLUSIVE, coords.region));
        launcher.add_region_requirement(RegionRequirement(p_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, p_vars.region));
        launcher.region_requirements[0].add_fields(field_id_coords);
        launcher.region_requirements[1].add_fields(field_id_p_vars);
        rt->execute_index_space(ctx, launcher);
        printf("Done!\n");
    }

    { // Launch taskConvertPrimitiveToConservative
        printf("Convert primitive variables to conservative variables ... ");
        ArgsConvertPrimitiveToConservative args_p2c;
        args_p2c.R_gas = args_solve.R_gas;
        args_p2c.gamma = args_solve.gamma;
        IndexLauncher launcher (
                TASK_ID::CVARS_TO_PVARS,
                color_isp_int,
                TaskArgument(&args_p2c, sizeof(ArgsConvertConservativeToPrimitive)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE,  p_vars.region));
        launcher.add_region_requirement(RegionRequirement(c0_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, c0_vars.region));
        launcher.region_requirements[0].add_fields(field_id_p_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
        printf("Done!\n");
    }

    printf("Starting time iterations:\n");
    for (int it = 0; it < args_solve.num_iter; it++) {
        launchSSPRK3(color_isp_int, color_isp_ghost_x, color_isp_ghost_y, c0_vars, c1_vars, c2_vars, p_vars, args_solve, ctx, rt);
        printf("  -- Iteration %04d done!\n", it);
    }

}



/*!
 * Generate 2D Cartesian mesh.
 *
 * Fields:
 *     [wo][0] COORD::X
 *     [wo][0] COORD::Y
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
    const Real R_gas    = args->R_gas;
    const Real gamm_inv = 1.0 / (args->gamma - 1.0);

    const PhysicalRegion& rgn_pvars = rgns[0];
    const PhysicalRegion& rgn_cvars = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> p_rho(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2> p_u  (rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2> p_v  (rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2> p_T  (rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rho (rgn_cvars, CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhou(rgn_cvars, CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_rhov(rgn_cvars, CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> c_Etot(rgn_cvars, CVARS_ID::ENRG);

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
    const Real R_gas  = args->R_gas;
    const Real gamm   = args->gamma - 1.0;
    const Real Rg_inv = 1.0 / R_gas;

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> c_rho (rgn_cvars, CVARS_ID::MASS);
    const FieldAccessor<READ_ONLY, Real, 2> c_rhou(rgn_cvars, CVARS_ID::MMTX);
    const FieldAccessor<READ_ONLY, Real, 2> c_rhov(rgn_cvars, CVARS_ID::MMTY);
    const FieldAccessor<READ_ONLY, Real, 2> c_Etot(rgn_cvars, CVARS_ID::ENRG);

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
 * [ro][0] PVARS::DENSITY
 * [ro][0] PVARS::VEL_X
 * [ro][0] PVARS::VEL_Y
 * [ro][0] PVARS::TEMP
 * [ro][1] CVARS::DENSITY
 * [ro][1] CVARS::VEL_X
 * [ro][1] CVARS::VEL_Y
 * [ro][1] CVARS::TEMP
 * [wo][2] CVARS::DENSITY
 * [wo][2] CVARS::VEL_X
 * [wo][2] CVARS::VEL_Y
 * [wo][2] CVARS::TEMP
 * Note: region[1] and region[2] are actually disjoint!
 *
 * Args: ArgsCalcRHS
 */
void taskCalcRHS(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    (void) ctx;

    auto args = reinterpret_cast<ArgsCalcRHS*>(task->args);
    const int STAGE_ID_NOW = args->stage_id_now;
    const int STENCIL_SIZE = args->stencil_size;
    const Real          Rg = args->R_gas;
    const Real         gam = args->gamma;
    const Real      mu_ref = args->mu_ref;
    const Real       T_ref = args->T_ref;
    const Real    visc_exp = args->visc_exp;
    const Real          Pr = args->Pr;
    const Real      dx_inv = 1.0 / args->dx;
    const Real      dy_inv = 1.0 / args->dy;
    const Real          dt = 1.0 / args->dt;

    const PhysicalRegion& rgn_pvars   = rgns[0];
    const PhysicalRegion& rgn_cvars_r = rgns[1];
    const PhysicalRegion& rgn_cvars_w = rgns[2];

    const FieldAccessor<READ_ONLY, Real, 2> rho_coll(rgn_pvars, PVARS_ID::DENSITY);
    const FieldAccessor<READ_ONLY, Real, 2>   u_coll(rgn_pvars, PVARS_ID::VEL_X  );
    const FieldAccessor<READ_ONLY, Real, 2>   v_coll(rgn_pvars, PVARS_ID::VEL_Y  );
    const FieldAccessor<READ_ONLY, Real, 2>   T_coll(rgn_pvars, PVARS_ID::TEMP   );

    const FieldAccessor<READ_ONLY, Real, 2> cons_mass(rgn_cvars_r, CVARS_ID::MASS);
    const FieldAccessor<READ_ONLY, Real, 2> cons_mmtx(rgn_cvars_r, CVARS_ID::MMTX);
    const FieldAccessor<READ_ONLY, Real, 2> cons_mmty(rgn_cvars_r, CVARS_ID::MMTY);
    const FieldAccessor<READ_ONLY, Real, 2> cons_enrg(rgn_cvars_r, CVARS_ID::ENRG);

    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mass(rgn_cvars_w, CVARS_ID::MASS);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmtx(rgn_cvars_w, CVARS_ID::MMTX);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_mmty(rgn_cvars_w, CVARS_ID::MMTY);
    const FieldAccessor<WRITE_DISCARD, Real, 2> ddt_enrg(rgn_cvars_w, CVARS_ID::ENRG);

    const int EDGE_STENCIL_SIZE = STENCIL_SIZE - 1;
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
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
            Point2D ije2 = ij + Point2D(idx_edge  , 0); 

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
            flx_enrg[idx_edge] = -rho * h * u + u * str11 + v * str21 + kap * gradT;
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
            flx_enrg[idx_edge] = -rho * h * v + u * str12 + v * str22 + kap * gradT;
        }
        const Real ddt_mass_y = ed04Stag(flx_mass[0], flx_mass[1], flx_mass[2], flx_mass[3], dy_inv);
        const Real ddt_mmtx_y = ed04Stag(flx_mmtx[0], flx_mmtx[1], flx_mmtx[2], flx_mmtx[3], dy_inv);
        const Real ddt_mmty_y = ed04Stag(flx_mmty[0], flx_mmty[1], flx_mmty[2], flx_mmty[3], dy_inv);
        const Real ddt_enrg_y = ed04Stag(flx_enrg[0], flx_enrg[1], flx_enrg[2], flx_enrg[3], dy_inv);

        ddt_mass[ij] = cons_mass[ij] + dt * (ddt_mass_x + ddt_mass_y);
        ddt_mmtx[ij] = cons_mmtx[ij] + dt * (ddt_mmtx_x + ddt_mmtx_y);
        ddt_mmty[ij] = cons_mmty[ij] + dt * (ddt_mmty_x + ddt_mmty_y);
        ddt_enrg[ij] = cons_enrg[ij] + dt * (ddt_enrg_x + ddt_enrg_y);
    }
}





/*!
 * Calculate the right-hand sides of evolution equations
 *
 * Fields:
 * [ro][0] CVARS::DENSITY
 * [ro][0] CVARS::VEL_X
 * [ro][0] CVARS::VEL_Y
 * [ro][0] CVARS::TEMP
 * [rw][1] CVARS::DENSITY
 * [rw][1] CVARS::VEL_X
 * [rw][1] CVARS::VEL_Y
 * [rw][1] CVARS::TEMP
 * Note region[0] and region[1] are actually disjoint!
 *
 * Args: ArgsCalcRHS
 */
void taskSSPRK3LinearCombination1(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    const PhysicalRegion& rgn_cvars_r = rgns[0];
    const PhysicalRegion& rgn_cvars_w = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> mass_0(rgn_cvars_r, CVARS_ID::MASS);
    const FieldAccessor<READ_ONLY, Real, 2> mmtx_0(rgn_cvars_r, CVARS_ID::MMTX);
    const FieldAccessor<READ_ONLY, Real, 2> mmty_0(rgn_cvars_r, CVARS_ID::MMTY);
    const FieldAccessor<READ_ONLY, Real, 2> enrg_0(rgn_cvars_r, CVARS_ID::ENRG);

    const FieldAccessor<READ_WRITE, Real, 2> mass_1(rgn_cvars_w, CVARS_ID::MASS);
    const FieldAccessor<READ_WRITE, Real, 2> mmtx_1(rgn_cvars_w, CVARS_ID::MMTX);
    const FieldAccessor<READ_WRITE, Real, 2> mmty_1(rgn_cvars_w, CVARS_ID::MMTY);
    const FieldAccessor<READ_WRITE, Real, 2> enrg_1(rgn_cvars_w, CVARS_ID::ENRG);

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        mass_1[ij] = mass_0[ij] * 0.75 + mass_1[ij] * 0.25;
        mmtx_1[ij] = mmtx_0[ij] * 0.75 + mmtx_1[ij] * 0.25;
        mass_1[ij] = mmty_0[ij] * 0.75 + mmty_1[ij] * 0.25;
        enrg_1[ij] = enrg_0[ij] * 0.75 + enrg_1[ij] * 0.25;
    }
}



/*!
 * Calculate the right-hand sides of evolution equations
 *
 * Fields:
 * [ro][0] CVARS::DENSITY
 * [ro][0] CVARS::VEL_X
 * [ro][0] CVARS::VEL_Y
 * [ro][0] CVARS::TEMP
 * [rw][1] CVARS::DENSITY
 * [rw][1] CVARS::VEL_X
 * [rw][1] CVARS::VEL_Y
 * [rw][1] CVARS::TEMP
 * Note region[0] and region[1] are actually disjoint!
 *
 * Args: ArgsCalcRHS
 */
void taskSSPRK3LinearCombination2(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    const PhysicalRegion& rgn_cvars_r = rgns[0];
    const PhysicalRegion& rgn_cvars_w = rgns[1];

    const FieldAccessor<READ_ONLY, Real, 2> mass_0(rgn_cvars_r, CVARS_ID::MASS);
    const FieldAccessor<READ_ONLY, Real, 2> mmtx_0(rgn_cvars_r, CVARS_ID::MMTX);
    const FieldAccessor<READ_ONLY, Real, 2> mmty_0(rgn_cvars_r, CVARS_ID::MMTY);
    const FieldAccessor<READ_ONLY, Real, 2> enrg_0(rgn_cvars_r, CVARS_ID::ENRG);

    const FieldAccessor<READ_WRITE, Real, 2> mass_1(rgn_cvars_w, CVARS_ID::MASS);
    const FieldAccessor<READ_WRITE, Real, 2> mmtx_1(rgn_cvars_w, CVARS_ID::MMTX);
    const FieldAccessor<READ_WRITE, Real, 2> mmty_1(rgn_cvars_w, CVARS_ID::MMTY);
    const FieldAccessor<READ_WRITE, Real, 2> enrg_1(rgn_cvars_w, CVARS_ID::ENRG);

    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (PointInBox2D pir(isp_domain); pir(); pir++) {
        Point2D ij = *pir;
        mass_1[ij] = mass_0[ij] * (1.0/3.0) + mass_1[ij] * (2.0/3.0);
        mmtx_1[ij] = mmtx_0[ij] * (1.0/3.0) + mmtx_1[ij] * (2.0/3.0);
        mass_1[ij] = mmty_0[ij] * (1.0/3.0) + mmty_1[ij] * (2.0/3.0);
        enrg_1[ij] = enrg_0[ij] * (1.0/3.0) + enrg_1[ij] * (2.0/3.0);
    }
}



/*
 * Function of launchers to conduct SSPRK3
 * Integrate resluts from t to t+dt
 * Task involved:
 *   taskConvertConservativeToPrimitive
 *   taskCalcRHS
 *   taskSSPRK3LinearCombination1
 *   taskSSPRK3LinearCombination2
 */
void launchSSPRK3(IndexSpace& color_isp_int, IndexSpace& color_isp_ghost_x, IndexSpace& color_isp_ghost_y,
        RegionOfFields& c0_vars, RegionOfFields& c1_vars, RegionOfFields& c2_vars, RegionOfFields& p_vars,
        ArgsSolve& args, Context ctx, Runtime* rt)
{
     
    /***************
      RegionOfFields is defined in util.h
        struct RegionOfFields {
            LogicalRegion    region;
            LogicalPartition patches_int;
            LogicalPartition patches_ext;
            LogicalPartition patches_ghost_x;
            LogicalPartition patches_ghost_y;
        };
    *****************/
    ArgsCalcRHS args_calcRHS;
    args_calcRHS.dt           = args.dt;
    args_calcRHS.dx           = args.dx;
    args_calcRHS.dy           = args.dy;
    args_calcRHS.R_gas        = args.R_gas;
    args_calcRHS.gamma        = args.gamma;
    args_calcRHS.stencil_size = args.stencil_size;

    ArgsConvertConservativeToPrimitive args_c2p;
    args_c2p.R_gas = args.R_gas;
    args_c2p.gamma = args.gamma;

    const std::vector<unsigned int> field_id_c_vars {0, 1, 2, 3};
    const std::vector<unsigned int> field_id_p_vars {0, 1, 2, 3};

    { // Launch taskConvertConservativeToPrimitive
        IndexLauncher launcher (
                TASK_ID::CVARS_TO_PVARS,
                color_isp_int,
                TaskArgument(&args_c2p, sizeof(ArgsConvertConservativeToPrimitive)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(c0_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c0_vars.region));
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE,  p_vars.region));
        launcher.region_requirements[0].add_fields(field_id_c_vars);
        launcher.region_requirements[1].add_fields(field_id_p_vars);
        rt->execute_index_space(ctx, launcher);
    }

    {// Fill up ghost points for periodic domain in p_vars
        IndexCopyLauncher launcher_x(color_isp_ghost_x);
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.src_requirements[0].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_x.src_requirements[1].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[1].add_fields(field_id_p_vars);

        IndexCopyLauncher launcher_y(color_isp_ghost_y);
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.src_requirements[0].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_y.src_requirements[1].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[1].add_fields(field_id_p_vars);
    }

    { // Launch calcRHS and write solutions to c1_vars
        IndexLauncher launcher (
                TASK_ID::CALC_RHS,
                color_isp_int,
                TaskArgument(&args_calcRHS, sizeof(ArgsCalcRHS)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_ext, 0, READ_ONLY,     EXCLUSIVE,  p_vars.region));
        launcher.add_region_requirement(RegionRequirement(c0_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c0_vars.region));
        launcher.add_region_requirement(RegionRequirement(c1_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, c1_vars.region));
        launcher.region_requirements[0].add_fields(field_id_p_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        launcher.region_requirements[2].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
    }

    { // Launch taskConvertConservativeToPrimitive
        IndexLauncher launcher (
                TASK_ID::CVARS_TO_PVARS,
                color_isp_int,
                TaskArgument(&args_c2p, sizeof(ArgsConvertConservativeToPrimitive)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(c1_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c1_vars.region));
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE,  p_vars.region));
        launcher.region_requirements[0].add_fields(field_id_c_vars);
        launcher.region_requirements[1].add_fields(field_id_p_vars);
        rt->execute_index_space(ctx, launcher);
    }

    {// Fill up ghost points for periodic domain in p_vars
        IndexCopyLauncher launcher_x(color_isp_ghost_x);
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.src_requirements[0].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_x.src_requirements[1].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[1].add_fields(field_id_p_vars);

        IndexCopyLauncher launcher_y(color_isp_ghost_y);
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.src_requirements[0].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_y.src_requirements[1].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[1].add_fields(field_id_p_vars);
    }

    { // Launch calcRHS and write solutions to c2_vars
        IndexLauncher launcher (
                TASK_ID::CALC_RHS,
                color_isp_int,
                TaskArgument(&args_calcRHS, sizeof(ArgsCalcRHS)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_ext, 0, READ_ONLY,     EXCLUSIVE,  p_vars.region));
        launcher.add_region_requirement(RegionRequirement(c1_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c1_vars.region));
        launcher.add_region_requirement(RegionRequirement(c2_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, c2_vars.region));
        launcher.region_requirements[0].add_fields(field_id_p_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        launcher.region_requirements[2].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
    }

    { // Launch taskSSPRK3LinearCombination1 and update c2 using c0
        IndexLauncher launcher (
                TASK_ID::SSPRK3_LINCOMB_1,
                color_isp_int,
                TaskArgument(),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(c0_vars.patches_int, 0, READ_ONLY,  EXCLUSIVE, c0_vars.region));
        launcher.add_region_requirement(RegionRequirement(c2_vars.patches_int, 0, READ_WRITE, EXCLUSIVE, c2_vars.region));
        launcher.region_requirements[0].add_fields(field_id_c_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
    }

    { // Launch taskConvertConservativeToPrimitive
        IndexLauncher launcher (
                TASK_ID::CVARS_TO_PVARS,
                color_isp_int,
                TaskArgument(&args_c2p, sizeof(ArgsConvertConservativeToPrimitive)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(c2_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c2_vars.region));
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE,  p_vars.region));
        launcher.region_requirements[0].add_fields(field_id_c_vars);
        launcher.region_requirements[1].add_fields(field_id_p_vars);
        rt->execute_index_space(ctx, launcher);
    }

    {// Fill up ghost points for periodic domain in p_vars
        IndexCopyLauncher launcher_x(color_isp_ghost_x);
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_x_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_x_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_x.src_requirements[0].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_x.src_requirements[1].add_fields(field_id_p_vars);
        launcher_x.dst_requirements[1].add_fields(field_id_p_vars);

        IndexCopyLauncher launcher_y(color_isp_ghost_y);
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_hi, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_lo,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.add_copy_requirements(
                RegionRequirement(p_vars.patches_ghost_mirror_y_lo, 0, READ_ONLY,     EXCLUSIVE, p_vars.region),
                RegionRequirement(p_vars.patches_ghost_y_hi,        0, WRITE_DISCARD, EXCLUSIVE, p_vars.region)
        );
        launcher_y.src_requirements[0].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[0].add_fields(field_id_p_vars);
        launcher_y.src_requirements[1].add_fields(field_id_p_vars);
        launcher_y.dst_requirements[1].add_fields(field_id_p_vars);
    }

    { // Launch calcRHS and write solutions to c1_vars
        IndexLauncher launcher (
                TASK_ID::CALC_RHS,
                color_isp_int,
                TaskArgument(&args_calcRHS, sizeof(ArgsCalcRHS)),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement( p_vars.patches_ext, 0, READ_ONLY,     EXCLUSIVE,  p_vars.region));
        launcher.add_region_requirement(RegionRequirement(c2_vars.patches_int, 0, READ_ONLY,     EXCLUSIVE, c2_vars.region));
        launcher.add_region_requirement(RegionRequirement(c1_vars.patches_int, 0, WRITE_DISCARD, EXCLUSIVE, c1_vars.region));
        launcher.region_requirements[0].add_fields(field_id_p_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        launcher.region_requirements[2].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
    }

    { // Launch taskSSPRK3LinearCombination2 and update c1 using c0
        IndexLauncher launcher (
                TASK_ID::SSPRK3_LINCOMB_2,
                color_isp_int,
                TaskArgument(),
                ArgumentMap()
        );
        launcher.add_region_requirement(RegionRequirement(c1_vars.patches_int, 0, READ_ONLY,  EXCLUSIVE, c1_vars.region));
        launcher.add_region_requirement(RegionRequirement(c0_vars.patches_int, 0, READ_WRITE, EXCLUSIVE, c0_vars.region));
        launcher.region_requirements[0].add_fields(field_id_c_vars);
        launcher.region_requirements[1].add_fields(field_id_c_vars);
        rt->execute_index_space(ctx, launcher);
    }
}



