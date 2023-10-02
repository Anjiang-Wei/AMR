import "regent"
local c      = regentlib.c
local format = require("std/format")
local grid   = require("grid")
require("fields")


task main()
    c.printf("Solver initialization:")
    c.printf("  -- Patch size (interior) : %d x %d\n", grid.patch_size, grid.patch_size)
    c.printf("  -- Ghost points on each side in each dimension: %d\n", grid.num_ghosts)
    c.printf("  -- Base level patches: %d x %d\n", grid.num_base_patches_i, grid.num_base_patches_j)
    c.printf("  -- Maximum level of refinement: %d\n", grid.level_max)
    c.printf("  -- Maximum possible allocated patches: %d\n", grid.num_patches_max)
    
    var color_space = ispace(int1d, grid.num_patches_max, 0)
    var patches_grid = region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), grid_fsp)
    var patches_meta = region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)

    var part_patches_meta             = grid.createPartitionOfMetaPatches(patches_meta) -- complete partition of patches_meta
    var part_patches_grid             = grid.createPartitionOfFullPatches(patches_grid) -- complete partition of patches_grid

    var part_patches_grid_int         = grid.createPartitionOfInteriorPatches (patches_grid)
    var part_patches_grid_i_prev_send = grid.createPartitionOfIPrevSendBuffers(patches_grid)
    var part_patches_grid_i_next_send = grid.createPartitionOfINextSendBuffers(patches_grid)
    var part_patches_grid_i_prev_recv = grid.createPartitionOfIPrevRecvBuffers(patches_grid)
    var part_patches_grid_i_next_recv = grid.createPartitionOfINextRecvBuffers(patches_grid)
    var part_patches_grid_j_prev_send = grid.createPartitionOfJPrevSendBuffers(patches_grid)
    var part_patches_grid_j_next_send = grid.createPartitionOfJNextSendBuffers(patches_grid)
    var part_patches_grid_j_prev_recv = grid.createPartitionOfJPrevRecvBuffers(patches_grid)
    var part_patches_grid_j_next_recv = grid.createPartitionOfJNextRecvBuffers(patches_grid)

    __demand(__index_launch)
    for color in color_space do
        grid.baseMetaGridInit(part_patches_meta[color])
    end
    
    
end


local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
