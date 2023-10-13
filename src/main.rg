import "regent"
local c      = regentlib.c
local format = require("std/format")
local grid   = require("grid")
require("fields")

fspace my_fsp {
    var1 : double,
    var2 : double,
}

task main()
    c.printf("Solver initialization:")
    c.printf("  -- Patch size (interior) : %d x %d\n", grid.patch_size, grid.patch_size)
    c.printf("  -- Ghost points on each side in each dimension: %d\n", grid.num_ghosts)
    c.printf("  -- Base level patches: %d x %d\n", grid.num_base_patches_i, grid.num_base_patches_j)
    c.printf("  -- Maximum level of refinement: %d\n", grid.level_max)
    c.printf("  -- Maximum possible allocated patches: %d\n", grid.num_patches_max)
    
    var color_space  = [grid.createColorSpace()] -- ispace(int1d, grid.num_patches_max, 0)
    var patches_grid = [grid.createDataRegion(grid_fsp)] -- region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), grid_fsp)
    var patches_meta = [grid.createMetaRegion()] -- region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)

    var part_patches_meta             = grid.createPartitionOfMetaPatches(patches_meta); -- complete partition of patches_meta
    var part_patches_grid             = [grid.createPartitionOfFullPatches(grid_fsp)](patches_grid); -- complete partition of patches_grid

    var part_patches_grid_int         = [grid.createPartitionOfInteriorPatches (grid_fsp)](patches_grid);
    var part_patches_grid_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_i_next_send = [grid.createPartitionOfINextSendBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_i_next_recv = [grid.createPartitionOfINextRecvBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_j_next_send = [grid.createPartitionOfJNextSendBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(grid_fsp)](patches_grid);
    var part_patches_grid_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(grid_fsp)](patches_grid);

    fill(patches_meta.{level, i_coord, j_coord, i_next, j_next, i_prev, j_prev, parent, child1, child2, child3, child4}, -1);

    __demand(__index_launch)
    for color in color_space do
        grid.baseMetaGridInit(part_patches_meta[color])
    end

    var isp_bounds_recv = part_patches_grid_i_prev_recv[int1d(0)].ispace.bounds;
    var isp_bounds_send = part_patches_grid_i_next_send[int1d(0)].ispace.bounds;
    -- c.printf("recv_buf = [(%d, %d) -- (%d, %d)]; send_buf = [(%d, %d) -- (%d, %d)]\n", isp_bounds_recv.lo.y, isp_bounds_recv.lo.z, isp_bounds_recv.hi.y, isp_bounds_recv.hi.z, isp_bounds_send.lo.y, isp_bounds_send.lo.z, isp_bounds_send.hi.y, isp_bounds_send.hi.z);
    --[grid.deepCopy2(grid_fsp)](part_patches_grid_i_prev_recv[0], part_patches_grid_i_next_send[2])
    --[grid.deepCopy (grid_fsp)](part_patches_grid_i_prev_recv[0], part_patches_grid_i_next_send[2])
    --[grid.deepCopy (grid_fsp)](part_patches_grid_i_prev_recv[2], part_patches_grid_i_next_send[0])

    [grid.fillGhosts(grid_fsp)](
      patches_meta,
      part_patches_meta,
      patches_grid,
      part_patches_grid_i_prev_send,
      part_patches_grid_i_next_send,
      part_patches_grid_j_prev_send,
      part_patches_grid_j_next_send,
      part_patches_grid_i_prev_recv,
      part_patches_grid_i_next_recv,
      part_patches_grid_j_prev_recv,
      part_patches_grid_j_next_recv
    );
    
    
end


local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
