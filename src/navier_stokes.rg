import "regent"
local usr_config     = require("input")
local problem_config = require("problem_config")
local c              = regentlib.c
local stdlib         = terralib.includec("stdlib.h")
local string         = terralib.includec("string.h")
local format         = require("std/format")
local grid           = require("grid")
local numerics       = require(usr_config.numerics_modules)
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
    var rgn_patches_grid = [grid.createDataRegion(grid_fsp)];

    var rgn_patches_pvars = [grid.createDataRegion(PVARS)];

    var rgn_patches_cvars_0 = [grid.createDataRegion(CVARS)];

    var rgn_patches_cvars_1 = [grid.createDataRegion(CVARS)];

    var rgn_patches_cvars_2 = [grid.createDataRegion(CVARS)];

    -- region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)
    var rgn_patches_meta = [grid.createMetaRegion()];


    -- CREATE PARTITIONS
    var patches_meta             = grid.createPartitionOfMetaPatches(rgn_patches_meta); -- complete partition of rgn_patches_meta
    var patches_grid             = [grid.createPartitionOfFullPatches(grid_fsp)](rgn_patches_grid); -- complete partition of rgn_patches_grid
    var patches_grid_int         = [grid.createPartitionOfInteriorPatches (grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_send = [grid.createPartitionOfINextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_recv = [grid.createPartitionOfINextRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_send = [grid.createPartitionOfJNextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(grid_fsp)](rgn_patches_grid);

    var patches_pvars             = [grid.createPartitionOfFullPatches(PVARS)](rgn_patches_pvars); -- complete partition of rgn_patches_pvars
    var patches_pvars_int         = [grid.createPartitionOfInteriorPatches (PVARS)](rgn_patches_pvars);
    var patches_pvars_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_i_next_send = [grid.createPartitionOfINextSendBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_i_next_recv = [grid.createPartitionOfINextRecvBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_j_next_send = [grid.createPartitionOfJNextSendBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(PVARS)](rgn_patches_pvars);
    var patches_pvars_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(PVARS)](rgn_patches_pvars);

    var patches_cvars_0             = [grid.createPartitionOfFullPatches(CVARS)](rgn_patches_cvars_0); -- complete partition of rgn_patches_cvars_0
    var patches_cvars_0_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_0);

    var patches_cvars_1             = [grid.createPartitionOfFullPatches(CVARS)](rgn_patches_cvars_1); -- complete partition of rgn_patches_cvars_1
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
    fill(rgn_patches_pvars.{rho, u, v, T, p}, 0.0);

    __demand(__index_launch)
    for color = 0, num_base_patches_i * num_base_patches_j do
        problem_config.setInitialCondition(patches_grid[color], patches_meta[color], patches_pvars[color])
    end
end


-- TODO: Add local task (or terra function) primitiveToConservative
-- TODO: Add local task (or terra function) conservativeToPrimitive
-- TODO: Add local task SSP-RK3


return solver
