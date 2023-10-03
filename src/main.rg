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

    fill(patches_meta.{level, i_coord, j_coord, i_next, j_next, i_prev, j_prev, parent, child1, child2, child3, child4}, -1)

    __demand(__index_launch)
    for color in color_space do
        grid.baseMetaGridInit(part_patches_meta[color])
    end

    -- var isp_bounds_recv = part_patches_grid_i_prev_recv[int1d(0)].ispace.bounds
    -- var isp_bounds_send = part_patches_grid_i_next_send[int1d(0)].ispace.bounds
    -- c.printf("recv_buf = [(%d, %d) -- (%d, %d)]; send_buf = [(%d, %d) -- (%d, %d)]\n", isp_bounds_recv.lo.y, isp_bounds_recv.lo.z, isp_bounds_recv.hi.y, isp_bounds_recv.hi.z, isp_bounds_send.lo.y, isp_bounds_send.lo.z, isp_bounds_send.hi.y, isp_bounds_send.hi.z);
    
    
end


local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
