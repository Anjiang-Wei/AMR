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
end


local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
