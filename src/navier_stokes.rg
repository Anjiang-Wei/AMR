import "regent"
local usr_config     = require("input")
local problem_config = require("problem_config")
local c              = regentlib.c
local stdlib         = terralib.includec("stdlib.h")
local string         = terralib.includec("string.h")
local format         = require("std/format")
local grid           = require("grid")
local numerics       = require(usr_config.numerics_modules)
local eos            = require("eos")
require("fields")

local domain_length_x         = usr_config.domain_length_x
local domain_length_y         = usr_config.domain_length_y
local domain_shift_x          = usr_config.domain_shift_x
local domain_shift_y          = usr_config.domain_shift_y 
local patch_size              = usr_config.patch_size
local num_base_patches_i      = usr_config.num_base_patches_i
local num_base_patches_j      = usr_config.num_base_patches_j

local num_grid_points_base_i  = num_base_patches_i * patch_size
local num_grid_points_base_j  = num_base_patches_j * patch_size

local solver = {}

local
terra pow2(e : int) : int
    return 1 << e
end


-- Calculate coordinate of grid points with given patch coordinate and level
task solver.setGridPointCoordinates(
    grid_patch : region(ispace(int3d), grid_fsp),
    meta_patch : region(ispace(int1d), grid_meta_fsp)
)
where
    reads (meta_patch.{i_coord, j_coord, level}),
    writes (grid_patch)
do
    var level_fact  = pow2(meta_patch[0].level)
    var dx : double = domain_length_x / (num_grid_points_base_i * level_fact)
    var dy : double = domain_length_y / (num_grid_points_base_j * level_fact)
    var x_start = meta_patch[0].i_coord * patch_size * dx + domain_shift_x;
    var y_start = meta_patch[0].j_coord * patch_size * dy + domain_shift_y;
    for cij in grid_patch.ispace do
        grid_patch[cij].x = x_start + dx * cij.y
        grid_patch[cij].y = y_start + dy * cij.z
    end
end


-- Calculate velocity gradient tensor at collocated points
local task calcGradVelColl(
    grad_vel_coll_patch : region(ispace(int3d), GRAD_VEL),      -- patch that only contains the interior region
    c_vars_now_patch    : region(ispace(int3d), CVARS),         -- patch including the halo layer on each side
    meta_patch          : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(grad_vel_coll_patch),
    reads(c_vars_now_patch.{mass, mmtx, mmty}, meta_patch)
do
    var level_fact  = pow2(meta_patch[0].level)    
    var inv_dx : double = 1.0 / (domain_length_x / (num_grid_points_base_i * level_fact));
    var inv_dy : double = 1.0 / (domain_length_y / (num_grid_points_base_j * level_fact));
    var stencil_buf_u : double[numerics.stencil_width];
    var stencil_buf_v : double[numerics.stencil_width];
    var stencil_idx_shift : int = (numerics.stencil_width - 1) / 2;
    for cij in grad_vel_coll_patch.ispace do
        for m = 0, numerics.stencil_width do
            var cij_shifted : int3d = int3d({cij.x, cij.y + m - stencil_idx_shift, cij.z});
            stencil_buf_u[m] = c_vars_now_patch[cij_shifted].mmtx / c_vars_now_patch[cij_shifted].mass;
            stencil_buf_v[m] = c_vars_now_patch[cij_shifted].mmty / c_vars_now_patch[cij_shifted].mass;
        end
        grad_vel_coll_patch[cij].dudx = numerics.der1Coll(stencil_buf_u[0], stencil_buf_u[1], stencil_buf_u[2], stencil_buf_u[3], stencil_buf_u[4], inv_dx);
        grad_vel_coll_patch[cij].dvdx = numerics.der1Coll(stencil_buf_v[0], stencil_buf_v[1], stencil_buf_v[2], stencil_buf_v[3], stencil_buf_v[4], inv_dx);

        for m = 0, numerics.stencil_width do
            var cij_shifted : int3d = int3d({cij.x, cij.y, cij.z + m - stencil_idx_shift});
            stencil_buf_u[m] = c_vars_now_patch[cij_shifted].mmtx / c_vars_now_patch[cij_shifted].mass;
            stencil_buf_v[m] = c_vars_now_patch[cij_shifted].mmty / c_vars_now_patch[cij_shifted].mass;
        end
        grad_vel_coll_patch[cij].dudy = numerics.der1Coll(stencil_buf_u[0], stencil_buf_u[1], stencil_buf_u[2], stencil_buf_u[3], stencil_buf_u[4], inv_dy);
        grad_vel_coll_patch[cij].dvdy = numerics.der1Coll(stencil_buf_v[0], stencil_buf_v[1], stencil_buf_v[2], stencil_buf_v[3], stencil_buf_v[4], inv_dy);
    end
end


local struct PrimitiveVars {
    u : double,
    v : double,
    T : double,
    p : double
}

local terra conservativeToPrimitive (rho : double, rho_u : double, rho_v : double, rho_e : double) : PrimitiveVars
    var pvars : PrimitiveVars;
    pvars.u = rho_u / rho;
    pvars.v = rho_v / rho;
    pvars.T = ((rho_e / rho) - 0.5 * (pvars.u * pvars.u + pvars.v * pvars.v)) / eos.Rg * (eos.gamma - 1.0);
    pvars.p = rho * eos.Rg * pvars.T;
    return pvars;
end 

-- Calculate the right-hand side of the NS equations
local task calcRHS(
    c_vars_ddt_patch : region(ispace(int3d), CVARS),     -- patch that only contains the interior region
    c_vars_now_patch : region(ispace(int3d), CVARS),     -- patch including the halo layer on each side
    grid_patch       : region(ispace(int3d), grid_fsp)   -- patch including the halo layer on each side
)
where
    writes(c_vars_ddt_patch),
    reads (c_vars_now_patch, grid_patch)
do

    var    u_coll : double [numerics.stencil_width_ext];
    var    v_coll : double [numerics.stencil_width_ext];
    var    T_coll : double [numerics.stencil_width_ext];
    var    p_coll : double [numerics.stencil_width    ];
    var dudx_coll : double [numerics.stencil_width    ];
    var dudy_coll : double [numerics.stencil_width    ];
    var dvdx_coll : double [numerics.stencil_width    ];
    var dvdy_coll : double [numerics.stencil_width    ];

    var flux_mass : double [numerics.stencil_width];
    var flux_mmtx : double [numerics.stencil_width];
    var flux_mmty : double [numerics.stencil_width];
    var flux_enrg : double [numerics.stencil_width];

    var stencil_ctr     : int = (numerics.stencil_width     - 1) / 2;
    var stencil_ext_ctr : int = (numerics.stencil_width_ext - 1) / 2;
    for cij in c_vars_ddt_patch.ispace do
        -----------------------
        -- ASSEMBLE X-FLUXES --
        -----------------------
        for i = 0, numerics.stencil_width_ext do
            var cij_shifted : int3d = int3d({cij.x, cij.y - stencil_ext_ctr + i, cij.z});
            var pvars : PrmitiveVars = conservativeToPrimitive(c_vars_now_patch[cij_shifted].mass, c_vars_now_patch[cij_shifted].mmtx, c_vars_now_patch[cij_shifted].mmty, c_vars_now_patch[cij_shifted].enrg);
            u_coll[i] = pvars.u;
            v_coll[i] = pvars.v;
            T_coll[i] = pvars.T;
            p_coll[i] = pvars.p;
        end

        -----------------------
        -- ASSEMBLE Y-FLUXES --
        -----------------------

    end

end


task solver.main()
    c.printf("Solver initialization:")
    c.printf("  -- Patch size (interior) : %d x %d\n", grid.patch_size, grid.patch_size)
    c.printf("  -- Ghost points on each side in each dimension: %d\n", grid.num_ghosts)
    c.printf("  -- Base level patches: %d x %d\n", grid.num_base_patches_i, grid.num_base_patches_j)
    c.printf("  -- Maximum level of refinement: %d\n", grid.level_max)
    c.printf("  -- Maximum possible allocated patches: %d\n", grid.num_patches_max)

    -- ispace(int1d, grid.num_patches_max, 0)
    var color_space  = [grid.createColorSpace()];

    -- region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), grid_fsp)
    var rgn_patches_grid     = [grid.createDataRegion(grid_fsp)];
    var rgn_patches_grad_vel = [grid.createDataRegion(GRAD_VEL)];
    var rgn_patches_cvars_0  = [grid.createDataRegion(CVARS)];
    var rgn_patches_cvars_1  = [grid.createDataRegion(CVARS)];
    var rgn_patches_cvars_2  = [grid.createDataRegion(CVARS)];

    -- region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)
    var rgn_patches_meta = [grid.createMetaRegion()];


    -- CREATE PARTITIONS
    var patches_meta             = grid.createPartitionOfMetaPatches(rgn_patches_meta); -- complete partition of rgn_patches_meta
    var patches_grid             = [grid.createPartitionOfFullPatches     (grid_fsp)](rgn_patches_grid); -- complete partition of rgn_patches_grid
    var patches_grid_int         = [grid.createPartitionOfInteriorPatches (grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_send = [grid.createPartitionOfINextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_recv = [grid.createPartitionOfINextRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_send = [grid.createPartitionOfJNextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(grid_fsp)](rgn_patches_grid);

    var patches_grad_vel             = [grid.createPartitionOfFullPatches     (GRAD_VEL)](rgn_patches_grad_vel); -- complete partition of rgn_patches_grad_vel
    var patches_grad_vel_int         = [grid.createPartitionOfInteriorPatches (GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_next_send = [grid.createPartitionOfINextSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_next_recv = [grid.createPartitionOfINextRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_next_send = [grid.createPartitionOfJNextSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);

    var patches_cvars_0             = [grid.createPartitionOfFullPatches     (CVARS)](rgn_patches_cvars_0); -- complete partition of rgn_patches_cvars_0
    var patches_cvars_0_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_0);

    var patches_cvars_1             = [grid.createPartitionOfFullPatches     (CVARS)](rgn_patches_cvars_1); -- complete partition of rgn_patches_cvars_1
    var patches_cvars_1_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_1);

    var patches_cvars_2             = [grid.createPartitionOfFullPatches(CVARS)](rgn_patches_cvars_2); -- complete partition of rgn_patches_cvars_2
    var patches_cvars_2_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_2);


    -- INITIALIZE DATA PATCHES
    fill(rgn_patches_grid.x, 0);
    fill(rgn_patches_grid.y, 0);

    -- INITIALIZE META PATCHES to avoid warnings (write_discard not supported)
    fill(rgn_patches_meta.{level, i_coord, j_coord, i_prev, i_next, j_prev, j_next, parent}, 0);
    fill(rgn_patches_meta.child, [terralib.constant(`arrayof(int, 0, 0, 0, 0))]);
    fill(rgn_patches_meta.{refine_req, coarsen_req}, false);
    grid.metaGridInit(rgn_patches_meta, patches_meta);

    __demand(__index_launch)
    for color = 0, num_base_patches_i * num_base_patches_j do
       solver.setGridPointCoordinates(patches_grid[color], patches_meta[color])
    end

    -- INITIALIZE PVARS to get rid of warnings (write_discard not supported)
    fill(rgn_patches_grad_vel.{dudx, dudy, dvdx, dvdy}, 0.0);

    __demand(__index_launch)
    for color = 0, num_base_patches_i * num_base_patches_j do
        problem_config.setInitialCondition(patches_grid[color], patches_meta[color], patches_cvars_0[color])
    end
end


-- TODO: Add local task (or terra function) primitiveToConservative
-- TODO: Add local task (or terra function) conservativeToPrimitive
-- TODO: Add local task SSP-RK3


return solver
