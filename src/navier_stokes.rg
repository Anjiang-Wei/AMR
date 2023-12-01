import "regent"
local usr_config = require("input")
local c          = regentlib.c
local stdlib     = terralib.includec("stdlib.h")
local string     = terralib.includec("string.h")
local format     = require("std/format")
local grid       = require("grid")
local numerics   = require(usr_config.numerics_modules)
require("fields")


local solver = {}

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


    -- INITIALIZE DATA PATCHES
    fill(rgn_patches_grid.x, 0);
    fill(rgn_patches_grid.y, 0);
end


return solver
