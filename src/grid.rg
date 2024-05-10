import "regent"
local usr_config = require("input")
local numerics   = require("numerics")
local format         = require("std/format")
local c          = regentlib.c
local math       = terralib.includec("math.h")


local grid = {}

grid.num_base_patches_i = usr_config.num_base_patches_i
grid.num_base_patches_j = usr_config.num_base_patches_j
grid.patch_size         = usr_config.patch_size
grid.level_max          = usr_config.level_max
grid.num_ghosts         = usr_config.num_ghosts
grid.num_patches_max    = math.ceil( (bit.lshift(4, 2*grid.level_max) - 1) * grid.num_base_patches_i * grid.num_base_patches_j / 3)

grid.full_patch_size = grid.patch_size + 2 * grid.num_ghosts -- patch size including interior region and ghost region
grid.idx_min         = -grid.num_ghosts                      -- starting index of the patch index space in each dimension including ghost region
grid.idx_max         = grid.patch_size - 1 + grid.num_ghosts -- last index of the patch index space in each dimension including ghost region



fspace grid_fsp {
    x            : double,
    y            : double,
}


fspace grid_meta_fsp {
    level         : int,    -- level of grid
    i_coord       : int,    -- patch coordinate in i-dimension
    j_coord       : int,    -- patch coordinate in j-dimension
    i_prev        : int,    -- patch id of the neighboring patch ahead in i-dimension
    i_next        : int,    -- patch id of the neighboring patch after in i-dimension
    j_prev        : int,    -- patch id of the neighboring patch ahead in j-dimension
    j_next        : int,    -- patch id of the neighboring patch after in j-dimension
    i_prev_j_prev : int,
    i_prev_j_next : int,
    i_next_j_prev : int,
    i_next_j_next : int,
    parent        : int,    -- patch id of the parent patch
    child         : int[4], -- patch id of all children patches
    --
    -- ^ j
    -- |
    -- |---------------------|
    -- | child[2] | child[3] |
    -- |----------+----------|
    -- | child[0] | child[1] |
    -- |---------------------| --> i
    --
    refine_req  : bool,
    coarsen_req : bool
}


-- Return color space
function grid.createColorSpace()
    return rexpr ispace(int1d, grid.num_patches_max, 0) end
end


-- Create a parent region of patches
-- fsp : field space
function grid.createDataRegion (fsp)
    return rexpr region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), fsp) end
end



-- Create meta-patch using the color space
function grid.createMetaRegion()
    return rexpr region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp) end
end


-- Get a patch from a partition
function _patch(part, pid)
    return rexpr part[pid][pid] end
end



-- Color the given region of meta-patches and return the partition by colors
__demand(__leaf, __inline)
task grid.createPartitionOfMetaPatches (
    meta_patches : region(ispace(int1d), grid_meta_fsp)
)
    var isp = meta_patches.ispace
    var num_colors = isp.bounds.hi + 1 -- assert isp.bounds.lo == 0
    var csp = ispace(int1d, num_colors, 0)
    var coloring = c.legion_domain_point_coloring_create()
    for color = 0, int(num_colors) do
        c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_1d( c.legion_rect_1d_t{lo = int1d(color):to_point(), hi = int1d(color):to_point()}  ))
    end
    var p = partition(disjoint, complete, meta_patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



-- Create the partition of the full patches (including interior and ghost entries)
-- Usage: var p = [grid.createPartitionOfFullPatches(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of rgn by its left-most dimension as the color index-space
function grid.createPartitionOfFullPatches(fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfFullPatchesTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var full_patch_bounds_lo = int3d({color, grid.idx_min, grid.idx_min})
            var full_patch_bounds_hi = int3d({color, grid.idx_max, grid.idx_max})
            var full_patch_bounds    = c.legion_rect_3d_t {lo = full_patch_bounds_lo:to_point(), hi = full_patch_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(full_patch_bounds))
        end
        var p = partition(disjoint, complete, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfFullPatchesTask
end



-- Create the partition of the interior patches
-- Usage: var p = [grid.createPartitionOfInteriorPatches(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of interior sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfInteriorPatches(fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfInteriorPatchesTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var int_patch_bounds_lo = int3d({color, 0, 0})
            var int_patch_bounds_hi = int3d({color, grid.patch_size-1, grid.patch_size-1})
            var int_patch_bounds    = c.legion_rect_3d_t {lo = int_patch_bounds_lo:to_point(), hi = int_patch_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(int_patch_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfInteriorPatchesTask
end



-- Create the partition of the send buffers in i-dimension
-- Usage: var p = [grid.createPartitionOfIPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevSendBuffers(fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevSendBuffersTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, 0})
            var ghost_bounds_hi = int3d({color, grid.num_ghosts-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevSendBuffersTask
end



-- Create the partition of the send buffers in i-dimension
-- Usage: var p = [grid.createPartitionOfINextSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextSendBuffersTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size-grid.num_ghosts, 0})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextSendBuffersTask
end



-- Create the partition of the receive buffers in i-dimension
-- Usage: var p = [grid.createPartitionOfIPrevRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevRecvBuffersTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, -grid.num_ghosts, 0})
            var ghost_bounds_hi = int3d({color, -1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevRecvBuffersTask
end



-- Create the partition of the receive buffers in i-dimension
-- Usage: var p = [grid.createPartitionOfINextRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextRecvBuffersTask (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size, 0})
            var ghost_bounds_hi = int3d({color, grid.patch_size+grid.num_ghosts-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextRecvBuffersTask
end



-- Create the partition of the send buffers in j-dimension
-- Usage: var p = [grid.createPartitionOfJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfJPrevSendBuffers (fsp)
local __demand(__leaf, __inline)
task createPartitionOfJPrevSendBuffersTask (
    patches : region(ispace(int3d), fsp) 
)
    var isp = patches.ispace
    var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
    var csp = ispace(int1d, num_colors, 0)

    var coloring = c.legion_domain_point_coloring_create()
    for color = 0, num_colors do
        var ghost_bounds_lo = int3d({color, 0, 0})
        var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.num_ghosts-1})
        var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
        c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
    end
    var p = partition(disjoint, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end
    return createPartitionOfJPrevSendBuffersTask
end



-- Create the partition of the send buffers in j-dimension
-- Usage: var p = [grid.createPartitionOfJNextSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfJNextSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfJNextSendBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, grid.patch_size-grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfJNextSendBuffers
end



-- Create the partition of the receive buffers in j-dimension
-- Usage: var p = [grid.createPartitionOfJPrevRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfJPrevRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfJPrevRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, -grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, -1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfJPrevRecvBuffers
end



-- Create the partition of the receive buffers in j-dimension
-- Usage: var p = [grid.createPartitionOfJNextRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfJNextRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfJNextRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, grid.patch_size})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.patch_size+grid.num_ghosts-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfJNextRecvBuffers
end



-- Create the partition of the send buffers of the i-prev j-prev corner
-- Usage: var p = [grid.createPartitionOfIPrevJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevJPrevSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevJPrevSendBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, 0})
            var ghost_bounds_hi = int3d({color, grid.num_ghosts-1, grid.num_ghosts-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevJPrevSendBuffers
end



-- Create the partition of the recv buffers of the i-prev j-prev corner
-- Usage: var p = [grid.createPartitionOfIPrevJPrevRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevJPrevRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevJPrevRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, -grid.num_ghosts, -grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, -1, -1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevJPrevRecvBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextJPrevSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextJPrevSendBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size-grid.num_ghosts, 0})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.num_ghosts-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextJPrevSendBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevRecvBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextJPrevRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextJPrevRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size, -grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, grid.patch_size+grid.num_ghosts-1, -1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextJPrevRecvBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevJNextSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevJNextSendBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, 0, grid.patch_size-grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, grid.num_ghosts-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevJNextSendBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfIPrevJNextRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfIPrevJNextRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, -grid.num_ghosts, grid.patch_size})
            var ghost_bounds_hi = int3d({color, -1, grid.patch_size+grid.num_ghosts-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfIPrevJNextRecvBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextJNextSendBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextJNextSendBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size-grid.num_ghosts, grid.patch_size-grid.num_ghosts})
            var ghost_bounds_hi = int3d({color, grid.patch_size-1, grid.patch_size-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextJNextSendBuffers
end



-- Create the partition of the send buffers of the i-next j-prev corner
-- Usage: var p = [grid.createPartitionOfINextJPrevSendBuffers(fsp)](rgn)
-- fsp : field space
-- rgn : parent region of data patches with fsp as the field space
-- p   : partition of the sub-rgn by its left-most dimension as the color index-space
function grid.createPartitionOfINextJNextRecvBuffers (fsp)
    local __demand(__leaf, __inline)
    task createPartitionOfINextJNextRecvBuffers (
        patches : region(ispace(int3d), fsp) 
    )
        var isp = patches.ispace
        var num_colors = isp.bounds.hi.x + 1 -- assert isp.bounds.lo.x == 0
        var csp = ispace(int1d, num_colors, 0)

        var coloring = c.legion_domain_point_coloring_create()
        for color = 0, num_colors do
            var ghost_bounds_lo = int3d({color, grid.patch_size, grid.patch_size})
            var ghost_bounds_hi = int3d({color, grid.patch_size+grid.num_ghosts-1, grid.patch_size+grid.num_ghosts-1})
            var ghost_bounds    = c.legion_rect_3d_t {lo = ghost_bounds_lo:to_point(), hi = ghost_bounds_hi:to_point()}
            c.legion_domain_point_coloring_color_domain(coloring, int1d(color):to_domain_point(), c.legion_domain_from_rect_3d(ghost_bounds))
        end
        var p = partition(disjoint, patches, coloring, csp)
        c.legion_domain_point_coloring_destroy(coloring)
        return p
    end
    return createPartitionOfINextJNextRecvBuffers
end



function grid.groupCommPartitions(FSP)
    local
    fspace parts(rgn : region(ispace(int3d),         FSP)) {
    i_prev_send        : partition(disjoint, rgn, ispace(int1d)),
    i_next_send        : partition(disjoint, rgn, ispace(int1d)),
    j_prev_send        : partition(disjoint, rgn, ispace(int1d)),
    j_next_send        : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_prev_send : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_next_send : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_prev_send : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_next_send : partition(disjoint, rgn, ispace(int1d)),
    i_prev_recv        : partition(disjoint, rgn, ispace(int1d)),
    i_next_recv        : partition(disjoint, rgn, ispace(int1d)),
    j_prev_recv        : partition(disjoint, rgn, ispace(int1d)),
    j_next_recv        : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_prev_recv : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_next_recv : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_prev_recv : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_next_recv : partition(disjoint, rgn, ispace(int1d))
}
    return parts
end

function grid.groupAllPartitions(FSP)
    local
    fspace parts(rgn : region(ispace(int3d),         FSP)) {
    patch_int          : partition(disjoint, rgn, ispace(int1d)),
    patch_full         : partition(disjoint, rgn, ispace(int1d)),
    i_prev_send        : partition(disjoint, rgn, ispace(int1d)),
    i_next_send        : partition(disjoint, rgn, ispace(int1d)),
    j_prev_send        : partition(disjoint, rgn, ispace(int1d)),
    j_next_send        : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_prev_send : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_next_send : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_prev_send : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_next_send : partition(disjoint, rgn, ispace(int1d)),
    i_prev_recv        : partition(disjoint, rgn, ispace(int1d)),
    i_next_recv        : partition(disjoint, rgn, ispace(int1d)),
    j_prev_recv        : partition(disjoint, rgn, ispace(int1d)),
    j_next_recv        : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_prev_recv : partition(disjoint, rgn, ispace(int1d)),
    i_prev_j_next_recv : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_prev_recv : partition(disjoint, rgn, ispace(int1d)),
    i_next_j_next_recv : partition(disjoint, rgn, ispace(int1d))
}
    return parts
end


-- Local helper function: calculate patch ID using the given patch coordinates
local
terra baseCoordToPid(i_coord : int, j_coord : int) : int
    if (i_coord < grid.num_base_patches_i and j_coord < grid.num_base_patches_j) then
        return i_coord * grid.num_base_patches_j + j_coord
    end
    return -1
end



-- Initialize the meta patches on base level
__demand(__leaf, __inline)
task grid.metaGridInit(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    writes (meta_patches_region)
do
    var num_base_patches : int = grid.num_base_patches_i * grid.num_base_patches_j
    for pid in meta_patches.colors do
        if (int(pid) < num_base_patches) then
            var my_i_coord : int = int(pid) / grid.num_base_patches_j
            var my_j_coord : int = int(pid) % grid.num_base_patches_j
            var nbr_i_coord_prev : int = int(my_i_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_i
            var nbr_i_coord_next : int = int(my_i_coord + 1                          ) % grid.num_base_patches_i
            var nbr_j_coord_prev : int = int(my_j_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_j
            var nbr_j_coord_next : int = int(my_j_coord + 1                          ) % grid.num_base_patches_j
            meta_patches[pid][pid].level          = 0
            meta_patches[pid][pid].i_coord        = my_i_coord
            meta_patches[pid][pid].j_coord        = my_j_coord
            meta_patches[pid][pid].i_prev         = baseCoordToPid(nbr_i_coord_prev,  my_j_coord)
            meta_patches[pid][pid].i_next         = baseCoordToPid(nbr_i_coord_next,  my_j_coord)
            meta_patches[pid][pid].j_prev         = baseCoordToPid( my_i_coord,      nbr_j_coord_prev)
            meta_patches[pid][pid].j_next         = baseCoordToPid( my_i_coord,      nbr_j_coord_next)
            meta_patches[pid][pid].i_prev_j_prev  = baseCoordToPid(nbr_i_coord_prev, nbr_j_coord_prev)
            meta_patches[pid][pid].i_prev_j_next  = baseCoordToPid(nbr_i_coord_prev, nbr_j_coord_next)
            meta_patches[pid][pid].i_next_j_prev  = baseCoordToPid(nbr_i_coord_next, nbr_j_coord_prev)
            meta_patches[pid][pid].i_next_j_next  = baseCoordToPid(nbr_i_coord_next, nbr_j_coord_next)
            --meta_patches[pid][pid].i_prev  = baseCoordToPid(int(my_i_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_i, my_j_coord)
            --meta_patches[pid][pid].i_next  = baseCoordToPid(int(my_i_coord + 1                          ) % grid.num_base_patches_i, my_j_coord)
            --meta_patches[pid][pid].j_prev  = baseCoordToPid(my_i_coord, int(my_j_coord - 1 + grid.num_base_patches_j) % grid.num_base_patches_j)
            --meta_patches[pid][pid].j_next  = baseCoordToPid(my_i_coord, int(my_j_coord + 1                          ) % grid.num_base_patches_j)
        else
            meta_patches[pid][pid].level          = -1
            meta_patches[pid][pid].i_coord        = -1
            meta_patches[pid][pid].j_coord        = -1
            meta_patches[pid][pid].i_prev         = -1
            meta_patches[pid][pid].i_next         = -1
            meta_patches[pid][pid].j_prev         = -1
            meta_patches[pid][pid].j_next         = -1
            meta_patches[pid][pid].i_prev_j_prev  = -1
            meta_patches[pid][pid].i_prev_j_next  = -1
            meta_patches[pid][pid].i_next_j_prev  = -1
            meta_patches[pid][pid].i_next_j_next  = -1
        end
        meta_patches[pid][pid].parent      = -1
        meta_patches[pid][pid].child[0]    = -1
        meta_patches[pid][pid].child[1]    = -1
        meta_patches[pid][pid].child[2]    = -1
        meta_patches[pid][pid].child[3]    = -1
        meta_patches[pid][pid].refine_req  = false
        meta_patches[pid][pid].coarsen_req = false
    end
end



-- Copy data in all fields between two (sub-) regions
-- Usage: [grid.deepCopy(fsp)](dst, src)
-- fsp : field space
-- dst : destination region
-- src : source region
-- Note: This function call assumes dst and src has the same size of index-space and they are disjoint (sub-) regions
function grid.deepCopy(fsp)
    local __demand(__leaf, __inline)
    task deepCopyTask(
        dst : region(ispace(int3d), fsp),
        src : region(ispace(int3d), fsp)
    )
    where
        reads(src), writes(dst)
    do
        var offset = src.bounds.lo - dst.bounds.lo
        for cij in dst.ispace do
            dst[cij] = src[cij+offset]
        end
    end
    return deepCopyTask
end


-- ------ Communication may occur
-- function grid.fillGhosts(fsp)
--     local
--     task taskFillGhosts (
--         meta_patches                    : region(ispace(int1d), grid_meta_fsp),
--         meta_patches_part               : partition(disjoint, meta_patches, ispace(int1d)),
--         data_patches                    : region(ispace(int3d), fsp),
--         data_patches_i_prev_send        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_send        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_j_prev_send        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_j_next_send        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_prev_recv        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_recv        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_j_prev_recv        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_j_next_recv        : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_prev_j_prev_send : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_prev_j_next_send : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_j_prev_send : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_j_next_send : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_prev_j_prev_recv : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_prev_j_next_recv : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_j_prev_recv : partition(disjoint, data_patches, ispace(int1d)),
--         data_patches_i_next_j_next_recv : partition(disjoint, data_patches, ispace(int1d))
--     )
--     where
--         reads (meta_patches.{i_prev, i_next, j_prev, j_next}),
--         reads writes (data_patches)
--     do
--         var csp = ispace(int1d, grid.num_patches_max, 0)
--         --__demand(__index_launch)
--         for pid in csp do
--             var i_prev : int = int(meta_patches_part[pid][pid].i_prev)
--             var i_next : int = int(meta_patches_part[pid][pid].i_next)
--             var j_prev : int = int(meta_patches_part[pid][pid].j_prev)
--             var j_next : int = int(meta_patches_part[pid][pid].j_next)

--             if (i_prev > -1) then
--                 [grid.deepCopy(fsp)](data_patches_i_next_recv[int1d(i_prev)], data_patches_i_prev_send[pid]);
--                 [grid.deepCopy(fsp)](data_patches_i_prev_recv[pid], data_patches_i_next_send[int1d(i_prev)]);
--             end
            
--             if (i_next > -1) then
--                 [grid.deepCopy(fsp)](data_patches_i_prev_recv[int1d(i_next)], data_patches_i_next_send[pid]);
--                 [grid.deepCopy(fsp)](data_patches_i_next_recv[pid], data_patches_i_prev_send[int1d(i_next)]);
--             end

--             if (j_prev > -1) then
--                 [grid.deepCopy(fsp)](data_patches_j_next_recv[int1d(j_prev)], data_patches_j_prev_send[pid]);
--                 [grid.deepCopy(fsp)](data_patches_j_prev_recv[pid], data_patches_j_next_send[int1d(j_prev)]);
--             end
            
--             if (j_next > -1) then
--                 [grid.deepCopy(fsp)](data_patches_j_prev_recv[int1d(j_next)], data_patches_j_next_send[pid]);
--                 [grid.deepCopy(fsp)](data_patches_j_next_recv[pid], data_patches_j_prev_send[int1d(j_next)]);
--             end

--         end
--     end

--     return taskFillGhosts
-- end


function grid.fillGhostsLevel(fsp, part_fsp)
    local __demand(__inline)
    task taskFillGhosts (
        level                    : int,
        meta_patches             : region(ispace(int1d), grid_meta_fsp),
        meta_patches_part        : partition(disjoint, meta_patches, ispace(int1d)),
        data_patches             : region(ispace(int3d), fsp),
        parts_patches            : part_fsp(data_patches)
    )
    where
        reads (meta_patches.{level, i_prev, i_next, j_prev, j_next, i_prev_j_prev, i_prev_j_next, i_next_j_prev, i_next_j_next}),
        reads writes (data_patches)
    do
        var csp = ispace(int1d, grid.num_patches_max, 0)
        --__demand(__index_launch)
        for pid in csp do
            if (level == meta_patches_part[pid][pid].level) then
                var i_prev        : int = int(meta_patches_part[pid][pid].i_prev)
                var i_next        : int = int(meta_patches_part[pid][pid].i_next)
                var j_prev        : int = int(meta_patches_part[pid][pid].j_prev)
                var j_next        : int = int(meta_patches_part[pid][pid].j_next)
                var i_prev_j_prev : int = int(meta_patches_part[pid][pid].i_prev_j_prev)
                var i_prev_j_next : int = int(meta_patches_part[pid][pid].i_prev_j_next)
                var i_next_j_prev : int = int(meta_patches_part[pid][pid].i_next_j_prev)
                var i_next_j_next : int = int(meta_patches_part[pid][pid].i_next_j_next)

                if (i_prev > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_next_recv[int1d(i_prev)], parts_patches.i_prev_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_prev_recv[pid], parts_patches.i_next_send[int1d(i_prev)]);
                end
                
                if (i_next > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_prev_recv[int1d(i_next)], parts_patches.i_next_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_next_recv[pid], parts_patches.i_prev_send[int1d(i_next)]);
                end

                if (j_prev > -1) then
                    [grid.deepCopy(fsp)](parts_patches.j_next_recv[int1d(j_prev)], parts_patches.j_prev_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.j_prev_recv[pid], parts_patches.j_next_send[int1d(j_prev)]);
                end
                
                if (j_next > -1) then
                    [grid.deepCopy(fsp)](parts_patches.j_prev_recv[int1d(j_next)], parts_patches.j_next_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.j_next_recv[pid], parts_patches.j_prev_send[int1d(j_next)]);
                end
                
                if (i_prev_j_prev > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_next_j_next_recv[int1d(i_prev_j_prev)], parts_patches.i_prev_j_prev_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_prev_j_prev_recv[pid], parts_patches.i_next_j_next_send[int1d(i_prev_j_prev)]);
                end

                if (i_prev_j_next > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_next_j_prev_recv[int1d(i_prev_j_next)], parts_patches.i_prev_j_next_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_prev_j_next_recv[pid], parts_patches.i_next_j_prev_send[int1d(i_prev_j_next)]);
                end

                if (i_next_j_prev > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_prev_j_next_recv[int1d(i_next_j_prev)], parts_patches.i_next_j_prev_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_next_j_prev_recv[pid], parts_patches.i_prev_j_next_send[int1d(i_next_j_prev)]);
                end

                if (i_next_j_next > -1) then
                    [grid.deepCopy(fsp)](parts_patches.i_prev_j_prev_recv[int1d(i_next_j_next)], parts_patches.i_next_j_next_send[pid]);
                    [grid.deepCopy(fsp)](parts_patches.i_next_j_next_recv[pid], parts_patches.i_prev_j_prev_send[int1d(i_next_j_next)]);
                end

            end
        end
    end

    return taskFillGhosts
end





-- Helper task: return the first pid wtih continuous available segment of "length" patches
local __demand(__leaf, __inline)
task getAvailableSegPatches(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d)),
    length              : int
)
where
    -- reads writes atomic(meta_patches_region.level)
    reads writes (meta_patches_region.level)
do
    var offset     : int = grid.num_base_patches_i * grid.num_base_patches_j
    var num_checks : int = grid.num_patches_max - offset - length + 1
    var valid : bool
    var ret: int = -1
    for pid in ispace(int1d, num_checks, offset) do
        valid = true
        for j = 0, length do
            var pid_next = pid + int1d(j)
            valid = valid and (meta_patches[pid_next][pid_next].level < 0)
        end
        if (valid) then
            for j = 0, length do
                var pid_next = pid + int1d(j)
                meta_patches[pid_next][pid_next].level = 9999
            end
            ret = pid
            break
        end
    end
    if (ret == -1) then
        regentlib.assert(false, "Cannot find any segment of available patches.")
    end
    return int1d(ret)
end


-- Enforce the activation of refine requests follows the AMR rule of refinement.
-- TODO: finish this for more complicated refine patterns
local __demand(__leaf, __inline)
task metaRefineFix(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
do
    for pid in meta_patches.colors do
        if (meta_patches[pid][pid].refine_req == true) then
            if (meta_patches[pid][pid].level < 0 or meta_patches[pid][pid].level == grid.level_max) then
                meta_patches[pid][pid].refine_req = false
            end
            if (meta_patches[pid][pid].child[0] > -1) then meta_patches[pid][pid].refine_req = false end
            if (meta_patches[pid][pid].i_prev   <  0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_prev)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].i_next   <  0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_next)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].j_prev   <  0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].j_prev)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].j_next   <  0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].j_next)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].i_next_j_next < 0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_next_j_next)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].i_prev_j_prev < 0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_prev_j_prev)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].i_prev_j_next < 0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_prev_j_next)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            if (meta_patches[pid][pid].i_next_j_prev < 0) then
                var pid_parent     : int1d = int1d(meta_patches[pid][pid].parent)
                var pid_parent_nbr : int1d = int1d(meta_patches[pid_parent][pid_parent].i_next_j_prev)
                meta_patches[pid_parent_nbr][pid_parent_nbr].refine_req = true
            end
            -- TODO: optimize this!!!!!!!!
        end
    end
end



local __demand(__leaf, __inline)
task metaCoarsenFix(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
do
    for pid in meta_patches.colors do
        if (meta_patches[pid][pid].coarsen_req == true) then
            if (meta_patches[pid][pid].level    <  1) then meta_patches[pid][pid].coarsen_req = false end
            if (meta_patches[pid][pid].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end

            var nbr_id : int;
            nbr_id = meta_patches[pid][pid].i_prev;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].j_prev;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].i_next;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].j_next;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end

            nbr_id = meta_patches[pid][pid].i_prev_j_prev;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].i_prev_j_next;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].i_next_j_prev;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
            nbr_id = meta_patches[pid][pid].i_next_j_next;
            if (nbr_id > -1 and meta_patches[int1d(nbr_id)][int1d(nbr_id)].child[0] > -1) then meta_patches[pid][pid].coarsen_req = false end
        end
    end
end



-- Calculate refined pattern using meta-patches only
--
-- ^ j
-- |
-- |---------------------|
-- | child[2] | child[3] |
-- |----------+----------|
-- | child[0] | child[1] |
-- |---------------------| --> i
--
__demand(__inline)
task grid.refineInit(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
    -- reads writes atomic(meta_patches_region)
do

    metaRefineFix(meta_patches_region, meta_patches)

    for pid in meta_patches.colors do
        var parent_patch = meta_patches[pid][pid];
        if ((parent_patch.level > -1) and (parent_patch.child[0] < 0) and parent_patch.refine_req) then
            -- Get four available patches from the list as the four new children
            var pid_child : int1d = getAvailableSegPatches(meta_patches_region, meta_patches, 4)
            for child_loc = 0, 4 do
                var pid_child_j : int1d = pid_child + child_loc
                meta_patches[pid_child_j][pid_child_j].level = parent_patch.level + 1
                meta_patches[pid_child_j][pid_child_j].parent = int(pid)
                meta_patches[pid_child_j][pid_child_j].i_coord = parent_patch.i_coord * 2 + int(child_loc == 1) + int(child_loc == 3)
                meta_patches[pid_child_j][pid_child_j].j_coord = parent_patch.j_coord * 2 + int(child_loc == 2) + int(child_loc == 3)
                meta_patches[pid][pid].child[child_loc] = int(pid_child_j)
            end -- for child_loc
        end
    end -- for pid

    for pid in meta_patches.colors do
        var parent_patch = meta_patches[pid][pid];
        if (parent_patch.refine_req) then
            for child_loc = 0, 4 do
                var pid_child_j : int1d = parent_patch.child[child_loc]

                if (child_loc == 0 or child_loc == 2) then
                    meta_patches[pid_child_j][pid_child_j].i_next = int(pid_child_j) + 1
                    if (parent_patch.i_prev > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_prev][parent_patch.i_prev]
                        var i_prev : int = parent_nbr.child[child_loc + 1]
                        meta_patches[pid_child_j][pid_child_j].i_prev = i_prev
                        if (i_prev > -1) then meta_patches[int1d(i_prev)][int1d(i_prev)].i_next = int(pid_child_j) end
                    else
                        meta_patches[pid_child_j][pid_child_j].i_prev = -1
                    end
                else
                    meta_patches[pid_child_j][pid_child_j].i_prev = int(pid_child_j) - 1
                    if (parent_patch.i_next > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_next][parent_patch.i_next]
                        var i_next : int = parent_nbr.child[child_loc - 1]
                        meta_patches[pid_child_j][pid_child_j].i_next = i_next
                        if (i_next > -1) then meta_patches[int1d(i_next)][int1d(i_next)].i_prev = int(pid_child_j) end
                    else
                        meta_patches[pid_child_j][pid_child_j].i_next = -1
                    end
                end

                if (child_loc == 0 or child_loc == 1) then
                    meta_patches[pid_child_j][pid_child_j].j_next = int(pid_child_j) + 2
                    if (parent_patch.j_prev > -1) then
                        var parent_nbr = meta_patches[parent_patch.j_prev][parent_patch.j_prev]
                        var j_prev : int = parent_nbr.child[child_loc + 2]
                        meta_patches[pid_child_j][pid_child_j].j_prev = j_prev
                        if (j_prev > -1) then meta_patches[int1d(j_prev)][int1d(j_prev)].j_next = int(pid_child_j) end
                    else
                        meta_patches[pid_child_j][pid_child_j].j_prev = -1
                    end
                else
                    meta_patches[pid_child_j][pid_child_j].j_prev = int(pid_child_j) - 2
                    if (parent_patch.j_next > -1) then
                        var parent_nbr = meta_patches[parent_patch.j_next][parent_patch.j_next]
                        var j_next : int = parent_nbr.child[child_loc - 2]
                        meta_patches[pid_child_j][pid_child_j].j_next = j_next
                        if (j_next > -1) then meta_patches[int1d(j_next)][int1d(j_next)].j_prev = int(pid_child_j) end
                    else
                        meta_patches[pid_child_j][pid_child_j].j_next = -1
                    end
                end

                if     (child_loc == 0) then
                    if (parent_patch.i_prev_j_prev > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_prev_j_prev][parent_patch.i_prev_j_prev]
                        var i_prev_j_prev : int = parent_nbr.child[3];
                        meta_patches[pid_child_j][pid_child_j].i_prev_j_prev = i_prev_j_prev;
                        if (i_prev_j_prev > -1) then meta_patches[int1d(i_prev_j_prev)][int1d(i_prev_j_prev)].i_next_j_next = int(pid_child_j) end 
                    else
                        meta_patches[pid_child_j][pid_child_j].i_prev_j_prev = -1;
                    end
                elseif (child_loc == 1) then
                    if (parent_patch.i_next_j_prev > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_next_j_prev][parent_patch.i_next_j_prev]
                        var i_next_j_prev : int = parent_nbr.child[2];
                        meta_patches[pid_child_j][pid_child_j].i_next_j_prev = i_next_j_prev;
                        if (i_next_j_prev > -1) then meta_patches[int1d(i_next_j_prev)][int1d(i_next_j_prev)].i_prev_j_next = int(pid_child_j) end 
                    else
                        meta_patches[pid_child_j][pid_child_j].i_next_j_prev = -1;
                    end
                elseif (child_loc == 2) then
                    if (parent_patch.i_prev_j_next > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_prev_j_next][parent_patch.i_prev_j_next]
                        var i_prev_j_next : int = parent_nbr.child[1];
                        meta_patches[pid_child_j][pid_child_j].i_prev_j_next = i_prev_j_next;
                        if (i_prev_j_next > -1) then meta_patches[int1d(i_prev_j_next)][int1d(i_prev_j_next)].i_next_j_prev = int(pid_child_j) end 
                    else
                        meta_patches[pid_child_j][pid_child_j].i_prev_j_next = -1;
                    end
                elseif (child_loc == 3) then
                    if (parent_patch.i_next_j_next > -1) then
                        var parent_nbr = meta_patches[parent_patch.i_next_j_next][parent_patch.i_next_j_next]
                        var i_next_j_next : int = parent_nbr.child[0];
                        meta_patches[pid_child_j][pid_child_j].i_next_j_next = i_next_j_next;
                        if (i_next_j_next > -1) then meta_patches[int1d(i_next_j_next)][int1d(i_next_j_next)].i_prev_j_prev = int(pid_child_j) end 
                    else
                        meta_patches[pid_child_j][pid_child_j].i_next_j_next = -1;
                    end
                end
            end -- for child_loc
        end
    end -- for pid
end



-- Clear all refine requests after the refinement is completed
__demand(__inline)
task grid.refineEnd(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(meta_patches_region.refine_req)
do
    fill(meta_patches_region.refine_req, false)
end



-- Helper function to test if all children of a given patch are leaves and they all have coarsen requests
-- Args:
--  parent_patch : sub-region of meta patch
--  meta_patches : partition of all meta patches
local function _allChildrenAreCoarsenable(parent_patch, meta_patches)
    return rexpr
        (parent_patch.level    > -1)                                              and
        (parent_patch.child[0] > -1)                                              and
        (meta_patches[parent_patch.child[0]][parent_patch.child[0]].child[0] < 0) and
        (meta_patches[parent_patch.child[1]][parent_patch.child[1]].child[0] < 0) and
        (meta_patches[parent_patch.child[2]][parent_patch.child[2]].child[0] < 0) and
        (meta_patches[parent_patch.child[3]][parent_patch.child[3]].child[0] < 0) and
        (meta_patches[parent_patch.child[0]][parent_patch.child[0]].coarsen_req)  and
        (meta_patches[parent_patch.child[1]][parent_patch.child[1]].coarsen_req)  and
        (meta_patches[parent_patch.child[2]][parent_patch.child[2]].coarsen_req)  and
        (meta_patches[parent_patch.child[3]][parent_patch.child[3]].coarsen_req)
    end
end




-- Reset a meta-patch used for deleting a meta-patch
-- pid -- patch ID to be deleted
local __demand(__inline) task _resetLeafMetaPatch(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d)),
    pid                 : int
)
where
    reads writes (meta_patches_region)
do
    var i_prev        : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_prev       )
    var i_next        : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_next       )
    var j_prev        : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].j_prev       )
    var j_next        : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].j_next       )
    var i_prev_j_prev : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_prev_j_prev)
    var i_prev_j_next : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_prev_j_next)
    var i_next_j_prev : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_next_j_prev)
    var i_next_j_next : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_next_j_next)
    
    var pid_parent : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].parent)
    var i_coord    : int   = meta_patches[pid][pid].i_coord
    var j_coord    : int   = meta_patches[pid][pid].j_coord
    var child_loc  : int   = (i_coord % 2) + (j_coord % 2) * 2

    if (int(i_prev)        > -1) then meta_patches[i_prev][i_prev].i_next = -1 end
    if (int(j_prev)        > -1) then meta_patches[j_prev][j_prev].j_next = -1 end
    if (int(i_next)        > -1) then meta_patches[i_next][i_next].i_prev = -1 end
    if (int(j_next)        > -1) then meta_patches[j_next][j_next].j_prev = -1 end
    if (int(i_prev_j_prev) > -1) then meta_patches[i_prev_j_prev][i_prev_j_prev].i_next_j_next = -1 end
    if (int(i_prev_j_next) > -1) then meta_patches[i_prev_j_next][i_prev_j_next].i_next_j_prev = -1 end
    if (int(i_next_j_prev) > -1) then meta_patches[i_next_j_prev][i_next_j_prev].i_prev_j_next = -1 end
    if (int(i_next_j_next) > -1) then meta_patches[i_next_j_next][i_next_j_next].i_prev_j_prev = -1 end
    
    meta_patches[pid][pid].level                          = -1;
    meta_patches[pid][pid].i_coord                        = -1;
    meta_patches[pid][pid].j_coord                        = -1;
    meta_patches[pid][pid].parent                         = -1;
    meta_patches[pid][pid].refine_req                     = false;
    meta_patches[pid][pid].coarsen_req                    = false;
    meta_patches[pid][pid].i_prev                         = -1;
    meta_patches[pid][pid].j_prev                         = -1;
    meta_patches[pid][pid].i_next                         = -1;
    meta_patches[pid][pid].j_next                         = -1;
    meta_patches[pid][pid].i_prev_j_prev                  = -1;
    meta_patches[pid][pid].i_prev_j_next                  = -1;
    meta_patches[pid][pid].i_next_j_prev                  = -1;
    meta_patches[pid][pid].i_next_j_next                  = -1;
    meta_patches[pid_parent][pid_parent].child[child_loc] = -1;
end


__demand(__inline)
task grid.coarsenInit(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
do
    metaCoarsenFix(meta_patches_region, meta_patches)
end



-- Calculate coarsened pattern using meta-patches only
--
-- ^ j
-- |
-- |---------------------|
-- | child[2] | child[3] |
-- |----------+----------|
-- | child[0] | child[1] |
-- |---------------------| --> i
--
__demand(__inline)
task grid.coarsenEnd(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
do
    for pid in meta_patches.colors do
        var parent_patch = meta_patches[pid][pid];
        if ([_allChildrenAreCoarsenable(parent_patch, meta_patches)]) then
            for child_loc = 0, 4 do
                var child_pid : int1d = parent_patch.child[child_loc];
                _resetLeafMetaPatch(meta_patches_region, meta_patches, child_pid)
            end -- for child_loc
        end
    end -- for pid
    fill(meta_patches_region.coarsen_req, false)
end



-- -- Upsample from parent to children
-- -- Parent needs to have valid ghost values
-- --
-- -- ^ j
-- -- |
-- -- |---------------------|
-- -- | child[2] | child[3] |
-- -- |----------+----------|
-- -- | child[0] | child[1] |
-- -- |---------------------| --> i
-- --
-- __demand(__inline)
-- task grid.upsample (
--     parent : region(ispace(int3d), double), -- full patch
--     child0 : region(ispace(int3d), double), -- interior patch
--     child1 : region(ispace(int3d), double), -- interior patch
--     child2 : region(ispace(int3d), double), -- interior patch
--     child3 : region(ispace(int3d), double)  -- interior patch
-- )
-- where
--     reads(parent),
--     writes(child0, child1, child2, child3)
-- do
--     -- cij means color-i-j
--     for cij in child0.ispace do
--         -- TODO
--         child0[cij] = parent[cij] -- placeholder scheme
--         child1[cij] = parent[cij] -- placeholder scheme
--         child2[cij] = parent[cij] -- placeholder scheme
--         child3[cij] = parent[cij] -- placeholder scheme
--     end
-- end



-- -- Downsample from children to parent
-- -- All children to have valid ghost values
-- --
-- -- ^ j
-- -- |
-- -- |---------------------|
-- -- | child[2] | child[3] |
-- -- |----------+----------|
-- -- | child[0] | child[1] |
-- -- |---------------------| --> i
-- --
-- __demand(__inline)
-- task grid.downsample (
--     child0 : region(ispace(int3d), double), -- full patch
--     child1 : region(ispace(int3d), double), -- full patch
--     child2 : region(ispace(int3d), double), -- full patch
--     child3 : region(ispace(int3d), double), -- full patch
--     parent : region(ispace(int3d), double)  -- interior patch
-- )
-- where
--     reads(child0, child1, child2, child3),
--     writes(parent)
-- do
--     -- cij means color-i-j
--     for cij in parent.ispace do
--         var child_i_loc : int = int(cij.y >= (grid.patch_size / 2))
--         var child_j_loc : int = int(cij.z >= (grid.patch_size / 2))
--         var child_loc   : int = child_i_loc + child_j_loc * 2
--         -- TODO
--         if     (child_loc == 0) then parent[cij] = child0[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
--         elseif (child_loc == 1) then parent[cij] = child1[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
--         elseif (child_loc == 2) then parent[cij] = child2[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
--         elseif (child_loc == 3) then parent[cij] = child3[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
--         end
--     end
-- end

-- Read from data patches, and set refine/coarsen flags in meta patches
task grid.setRefineCoarsenFlags(
    patches_grid_region : region(ispace(int3d), grid_fsp),
    patches_grid        : partition(disjoint, complete, patches_grid_region, ispace(int1d)),
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads (patches_grid_region),
    writes (meta_patches_region.{refine_req, coarsen_req})
do

end




-- Upsample from parent to children
-- Parent needs to have valid ghost values
--
-- ^ j
-- |
-- |---------------------|
-- | child[2] | child[3] |
-- |----------+----------|
-- | child[0] | child[1] |
-- |---------------------| --> i
--
__demand(__leaf, __inline)
task grid.upsample (
    parent : region(ispace(int3d), double), -- full patch
    child0 : region(ispace(int3d), double), -- interior patch
    child1 : region(ispace(int3d), double), -- interior patch
    child2 : region(ispace(int3d), double), -- interior patch
    child3 : region(ispace(int3d), double)  -- interior patch
)
where
    reads(parent),
    writes(child0, child1, child2, child3)
do
    var parent_pid : int = int(parent.ispace.bounds.lo.x)
    var parent_int_isp_bounds_lo : int3d = int3d({parent_pid, child0.ispace.bounds.lo.y, child0.ispace.bounds.lo.z})
    var parent_int_isp_bounds_hi : int3d = int3d({parent_pid, child0.ispace.bounds.hi.y, child0.ispace.bounds.hi.z})
    var isp_int = ispace(int3d, parent_int_isp_bounds_hi - parent_int_isp_bounds_lo + int3d({1, 1, 1}), parent_int_isp_bounds_lo)
    var cij_0  : int3d = (isp_int.bounds.lo + isp_int.bounds.hi) / int3d({2, 2, 2})
    -- format.println("parent_pid: {}, isp_int : {}, cij_0 : {}", parent_pid, isp_int.bounds, cij_0)
    for cij in isp_int do
        var uL_jm2 : double = numerics.upSample(parent[cij+int3d({0, -2,-2})], parent[cij+int3d({0, -1,-2})], parent[cij], parent[cij+int3d({0,  1,-2})], parent[cij+int3d({0,  2,-2})])
        var uR_jm2 : double = numerics.upSample(parent[cij+int3d({0,  2,-2})], parent[cij+int3d({0,  1,-2})], parent[cij], parent[cij+int3d({0, -1,-2})], parent[cij+int3d({0, -2,-2})])
        var uL_jm1 : double = numerics.upSample(parent[cij+int3d({0, -2,-1})], parent[cij+int3d({0, -1,-1})], parent[cij], parent[cij+int3d({0,  1,-1})], parent[cij+int3d({0,  2,-1})])
        var uR_jm1 : double = numerics.upSample(parent[cij+int3d({0,  2,-1})], parent[cij+int3d({0,  1,-1})], parent[cij], parent[cij+int3d({0, -1,-1})], parent[cij+int3d({0, -2,-1})])
        var uL_j00 : double = numerics.upSample(parent[cij+int3d({0, -2, 0})], parent[cij+int3d({0, -1, 0})], parent[cij], parent[cij+int3d({0,  1, 0})], parent[cij+int3d({0,  2, 0})])
        var uR_j00 : double = numerics.upSample(parent[cij+int3d({0,  2, 0})], parent[cij+int3d({0,  1, 0})], parent[cij], parent[cij+int3d({0, -1, 0})], parent[cij+int3d({0, -2, 0})])
        var uL_jp1 : double = numerics.upSample(parent[cij+int3d({0, -2, 1})], parent[cij+int3d({0, -1, 1})], parent[cij], parent[cij+int3d({0,  1, 1})], parent[cij+int3d({0,  2, 1})])
        var uR_jp1 : double = numerics.upSample(parent[cij+int3d({0,  2, 1})], parent[cij+int3d({0,  1, 1})], parent[cij], parent[cij+int3d({0, -1, 1})], parent[cij+int3d({0, -2, 1})])
        var uL_jp2 : double = numerics.upSample(parent[cij+int3d({0, -2, 2})], parent[cij+int3d({0, -1, 2})], parent[cij], parent[cij+int3d({0,  1, 2})], parent[cij+int3d({0,  2, 2})])
        var uR_jp2 : double = numerics.upSample(parent[cij+int3d({0,  2, 2})], parent[cij+int3d({0,  1, 2})], parent[cij], parent[cij+int3d({0, -1, 2})], parent[cij+int3d({0, -2, 2})])

        var uL_im2 : double = numerics.upSample(parent[cij+int3d({0,-2, -2})], parent[cij+int3d({0,-2, -1})], parent[cij], parent[cij+int3d({0,-2,  1})], parent[cij+int3d({0,-2,  2})])
        var uR_im2 : double = numerics.upSample(parent[cij+int3d({0,-2,  2})], parent[cij+int3d({0,-2,  1})], parent[cij], parent[cij+int3d({0,-2, -1})], parent[cij+int3d({0,-2, -2})])
        var uL_im1 : double = numerics.upSample(parent[cij+int3d({0,-1, -2})], parent[cij+int3d({0,-1, -1})], parent[cij], parent[cij+int3d({0,-1,  1})], parent[cij+int3d({0,-1,  2})])
        var uR_im1 : double = numerics.upSample(parent[cij+int3d({0,-1,  2})], parent[cij+int3d({0,-1,  1})], parent[cij], parent[cij+int3d({0,-1, -1})], parent[cij+int3d({0,-1, -2})])
        var uL_i00 : double = numerics.upSample(parent[cij+int3d({0, 0, -2})], parent[cij+int3d({0, 0, -1})], parent[cij], parent[cij+int3d({0, 0,  1})], parent[cij+int3d({0, 0,  2})])
        var uR_i00 : double = numerics.upSample(parent[cij+int3d({0, 0,  2})], parent[cij+int3d({0, 0,  1})], parent[cij], parent[cij+int3d({0, 0, -1})], parent[cij+int3d({0, 0, -2})])
        var uL_ip1 : double = numerics.upSample(parent[cij+int3d({0, 1, -2})], parent[cij+int3d({0, 1, -1})], parent[cij], parent[cij+int3d({0, 1,  1})], parent[cij+int3d({0, 1,  2})])
        var uR_ip1 : double = numerics.upSample(parent[cij+int3d({0, 1,  2})], parent[cij+int3d({0, 1,  1})], parent[cij], parent[cij+int3d({0, 1, -1})], parent[cij+int3d({0, 1, -2})])
        var uL_ip2 : double = numerics.upSample(parent[cij+int3d({0, 2, -2})], parent[cij+int3d({0, 2, -1})], parent[cij], parent[cij+int3d({0, 2,  1})], parent[cij+int3d({0, 2,  2})])
        var uR_ip2 : double = numerics.upSample(parent[cij+int3d({0, 2,  2})], parent[cij+int3d({0, 2,  1})], parent[cij], parent[cij+int3d({0, 2, -1})], parent[cij+int3d({0, 2, -2})])
        --  ---------
        -- | LR | RR |
        -- |----|----|
        -- | LL | RL |
        --  ---------
        var uLL : double = (0.5 * numerics.upSample(uL_jm2, uL_jm1, uL_j00, uL_jp1, uL_jp2) + 0.5 * numerics.upSample(uL_im2, uL_im1, uL_i00, uL_ip1, uL_ip2))
        var uLR : double = (0.5 * numerics.upSample(uL_jp2, uL_jp1, uL_j00, uL_jm1, uL_jm2) + 0.5 * numerics.upSample(uR_im2, uR_im1, uR_i00, uR_ip1, uR_ip2))
        var uRL : double = (0.5 * numerics.upSample(uR_jm2, uR_jm1, uR_j00, uR_jp1, uR_jp2) + 0.5 * numerics.upSample(uL_ip2, uL_ip1, uL_i00, uL_im1, uL_im2))
        var uRR : double = (0.5 * numerics.upSample(uR_jp2, uR_jp1, uR_j00, uR_jm1, uR_jm2) + 0.5 * numerics.upSample(uR_ip2, uR_ip1, uR_i00, uR_im1, uR_im2))
        if     (cij.y <= cij_0.y and cij.z <= cij_0.z) then -- child[0]
            var pid : int = child0.bounds.lo.x
            var i_par : int = cij.y
            var j_par : int = cij.z
            var child_cij_LL : int3d = int3d({pid, i_par*2    , j_par*2    })
            var child_cij_LR : int3d = int3d({pid, i_par*2    , j_par*2 + 1})
            var child_cij_RL : int3d = int3d({pid, i_par*2 + 1, j_par*2    })
            var child_cij_RR : int3d = int3d({pid, i_par*2 + 1, j_par*2 + 1})
            child0[child_cij_LL] = uLL
            child0[child_cij_LR] = uLR
            child0[child_cij_RL] = uRL
            child0[child_cij_RR] = uRR

        elseif (cij.y >  cij_0.y and cij.z <= cij_0.z) then -- child[1]
            var pid : int = child1.bounds.lo.x
            var i_par : int = cij.y - (cij_0.y + 1)
            var j_par : int = cij.z
            var child_cij_LL : int3d = int3d({pid, i_par*2    , j_par*2    })
            var child_cij_LR : int3d = int3d({pid, i_par*2    , j_par*2 + 1})
            var child_cij_RL : int3d = int3d({pid, i_par*2 + 1, j_par*2    })
            var child_cij_RR : int3d = int3d({pid, i_par*2 + 1, j_par*2 + 1})
            child1[child_cij_LL] = uLL
            child1[child_cij_LR] = uLR
            child1[child_cij_RL] = uRL
            child1[child_cij_RR] = uRR

        elseif (cij.y <= cij_0.y and cij.z >  cij_0.z) then -- child[2]
            var pid : int = child2.bounds.lo.x
            var i_par : int = cij.y
            var j_par : int = cij.z - (cij_0.z + 1)
            var child_cij_LL : int3d = int3d({pid, i_par*2    , j_par*2    })
            var child_cij_LR : int3d = int3d({pid, i_par*2    , j_par*2 + 1})
            var child_cij_RL : int3d = int3d({pid, i_par*2 + 1, j_par*2    })
            var child_cij_RR : int3d = int3d({pid, i_par*2 + 1, j_par*2 + 1})
            child2[child_cij_LL] = uLL
            child2[child_cij_LR] = uLR
            child2[child_cij_RL] = uRL
            child2[child_cij_RR] = uRR

        elseif (cij.y >  cij_0.y and cij.z >  cij_0.z) then -- child[3]
            var pid : int = child3.bounds.lo.x
            var i_par : int = cij.y - (cij_0.y + 1)
            var j_par : int = cij.z - (cij_0.z + 1)
            var child_cij_LL : int3d = int3d({pid, i_par*2    , j_par*2    })
            var child_cij_LR : int3d = int3d({pid, i_par*2    , j_par*2 + 1})
            var child_cij_RL : int3d = int3d({pid, i_par*2 + 1, j_par*2    })
            var child_cij_RR : int3d = int3d({pid, i_par*2 + 1, j_par*2 + 1})
            child3[child_cij_LL] = uLL
            child3[child_cij_LR] = uLR
            child3[child_cij_RL] = uRL
            child3[child_cij_RR] = uRR

        end
    end
    ---- cij means color-i-j
    --for cij in child0.ispace do
    --    -- TODO
    --    child0[cij] = parent[cij] -- placeholder scheme
    --    child1[cij] = parent[cij] -- placeholder scheme
    --    child2[cij] = parent[cij] -- placeholder scheme
    --    child3[cij] = parent[cij] -- placeholder scheme
    --end
end



-- Downsample from children to parent
-- All children to have valid ghost values
--
-- ^ j
-- |
-- |---------------------|
-- | child[2] | child[3] |
-- |----------+----------|
-- | child[0] | child[1] |
-- |---------------------| --> i
--
__demand(__leaf, __inline)
task grid.downsample (
    child0 : region(ispace(int3d), double), -- full patch
    child1 : region(ispace(int3d), double), -- full patch
    child2 : region(ispace(int3d), double), -- full patch
    child3 : region(ispace(int3d), double), -- full patch
    parent : region(ispace(int3d), double)  -- interior patch
)
where
    reads(child0, child1, child2, child3),
    writes(parent)
do
    -- cij means color-i-j
    for cij in parent.ispace do
        var child_i_loc : int = int(cij.y >= (grid.patch_size / 2))
        var child_j_loc : int = int(cij.z >= (grid.patch_size / 2))
        var child_loc   : int = child_i_loc + child_j_loc * 2
        -- TODO
        if     (child_loc == 0) then parent[cij] = child0[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
        elseif (child_loc == 1) then parent[cij] = child1[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
        elseif (child_loc == 2) then parent[cij] = child2[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
        elseif (child_loc == 3) then parent[cij] = child3[int3d({cij.x, cij.y, cij.z})] -- placeholder scheme
        end
    end
end


return grid
