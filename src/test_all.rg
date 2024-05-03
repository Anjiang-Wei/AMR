import "regent"
local c      = regentlib.c
local stdlib = terralib.includec("stdlib.h")
local string = terralib.includec("string.h")
local format = require("std/format")
local grid   = require("grid")
require("fields")

fspace my_fsp {
    var1 : double,
    var2 : double,
}



-- This task writes meta-patch info of all active meta-patches
function writeActiveMeta(fname)
    local __demand(__inline) task taskWriteActiveMeta(
        meta_patches_region : region(ispace(int1d), grid_meta_fsp),
        meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
    )
    where
        reads (meta_patches_region)
    do
        var file = c.fopen(fname, "w")
        c.fprintf(file, "%7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s\n",
                    "pid", "level", "i_coord", "j_coord", "i_prev", "i_next", "j_prev", "j_next", "parent", "child0", "child1", "child2", "child3", "r_req", "c_req");
        for pid in meta_patches.colors do
            var patch = meta_patches[pid][pid]
            if (patch.level > -1) then
                c.fprintf(file, "%7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d\n",
                    int(pid), patch.level, patch.i_coord, patch.j_coord,
                    patch.i_prev, patch.i_next, patch.j_prev, patch.j_next,
                    patch.parent, patch.child[0], patch.child[1], patch.child[2], patch.child[3],
                    patch.refine_req, patch.coarsen_req);
            end
        end -- for pid
        c.fclose(file)
    end -- task
    return taskWriteActiveMeta
end


task main()
    var output_path = "./build"


    c.printf("Solver initialization:")
    c.printf("  -- Patch size (interior) : %d x %d\n", grid.patch_size, grid.patch_size)
    c.printf("  -- Ghost points on each side in each dimension: %d\n", grid.num_ghosts)
    c.printf("  -- Base level patches: %d x %d\n", grid.num_base_patches_i, grid.num_base_patches_j)
    c.printf("  -- Maximum level of refinement: %d\n", grid.level_max)
    c.printf("  -- Maximum possible allocated patches: %d\n", grid.num_patches_max)
    
    var color_space  = [grid.createColorSpace()]; -- ispace(int1d, grid.num_patches_max, 0)
    var patches_grid = [grid.createDataRegion(grid_fsp)]; -- region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), grid_fsp)
    var patches_meta = [grid.createMetaRegion()]; -- region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)

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


    -- TEST BASE GRID INITIALIZATION
    grid.metaGridInit(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_init.dat")](patches_meta, part_patches_meta);

    -- INITIALIZE DATA PATCHES
    fill(patches_grid.x, 0);
    fill(patches_grid.y, 0);
    -- SET REFINE/COARSEN FLAGS
    grid.setRefineCoarsenFlags(patches_grid, part_patches_grid_int, patches_meta, part_patches_meta);

    -- TEST REFINEMENT
    for pid in part_patches_meta.colors do
        part_patches_meta[pid][pid].refine_req = (stdlib.rand() % 2) == 1
    end
    [writeActiveMeta("output_meta_refine_stage_0.dat")](patches_meta, part_patches_meta);
    grid.refineInit(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_refine_stage_1.dat")](patches_meta, part_patches_meta);
    grid.refineEnd(patches_meta);
    [writeActiveMeta("output_meta_refine_stage_2.dat")](patches_meta, part_patches_meta);

    -- TEST COARSENING
    for pid in part_patches_meta.colors do
        part_patches_meta[pid][pid].coarsen_req = (stdlib.rand() % 6) > 0
    end
    [writeActiveMeta("output_meta_coarsen_stage_0.dat")](patches_meta, part_patches_meta);
    grid.coarsenInit(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_coarsen_stage_1.dat")](patches_meta, part_patches_meta);
    grid.coarsenEnd(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_coarsen_stage_2.dat")](patches_meta, part_patches_meta);
    
    -- TEST REFINEMENT AGAIN
    for pid in part_patches_meta.colors do
        part_patches_meta[pid][pid].refine_req = (stdlib.rand() % 3) == 0
    end
    [writeActiveMeta("output_meta_further_refine_stage_0.dat")](patches_meta, part_patches_meta);
    grid.refineInit(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_further_refine_stage_1.dat")](patches_meta, part_patches_meta);
    grid.refineEnd(patches_meta);
    [writeActiveMeta("output_meta_further_refine_stage_2.dat")](patches_meta, part_patches_meta);

    -- TEST COARSENING
    for pid in part_patches_meta.colors do
        part_patches_meta[pid][pid].coarsen_req = (stdlib.rand() % 4) > 0
    end
    [writeActiveMeta("output_meta_further_coarsen_stage_0.dat")](patches_meta, part_patches_meta);
    grid.coarsenInit(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_further_coarsen_stage_1.dat")](patches_meta, part_patches_meta);
    grid.coarsenEnd(patches_meta, part_patches_meta);
    [writeActiveMeta("output_meta_further_coarsen_stage_2.dat")](patches_meta, part_patches_meta);
    

    
end


local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
