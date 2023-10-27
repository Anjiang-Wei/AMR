import "regent"
local c    = regentlib.c
local math = terralib.includec("math.h")


local grid = {}


if grid.num_base_patches_i == nil then grid.num_base_patches_i = 8 end
if grid.num_base_patches_j == nil then grid.num_base_patches_j = 8 end
if grid.patch_size         == nil then grid.patch_size = 16 end
if grid.level_max          == nil then grid.level_max = 2 end
if grid.num_ghosts         == nil then grid.num_ghosts = 4 end
if grid.num_patches_max    == nil then grid.num_patches_max = math.ceil( (bit.lshift(4, 2*grid.level_max) - 1) * grid.num_base_patches_i * grid.num_base_patches_j / 3)  end

grid.full_patch_size = grid.patch_size + 2 * grid.num_ghosts -- patch size including interior region and ghost region
grid.idx_min         = -grid.num_ghosts                      -- starting index of the patch index space in each dimension including ghost region
grid.idx_max         = grid.patch_size - 1 + grid.num_ghosts -- last index of the patch index space in each dimension including ghost region



fspace grid_fsp {
    x            : double,
    y            : double,
}


fspace grid_meta_fsp {
    level   : int,    -- level of grid
    i_coord : int,    -- patch coordinate in i-dimension
    j_coord : int,    -- patch coordinate in j-dimension
    i_prev  : int,    -- patch id of the neighboring patch ahead in i-dimension
    i_next  : int,    -- patch id of the neighboring patch after in i-dimension
    j_prev  : int,    -- patch id of the neighboring patch ahead in j-dimension
    j_next  : int,    -- patch id of the neighboring patch after in j-dimension
    parent  : int,    -- patch id of the parent patch
    child   : int[4], -- patch id of all children patches
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
    local
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
    local
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
    local
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
    local
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
    local
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
    local
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
local
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
    local
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
    local
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
    local
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



-- Local helper function: calculate patch ID using the given patch coordinates
local
terra baseCoordToPid(i_coord : int, j_coord : int) : int
    if (i_coord < grid.num_base_patches_i and j_coord < grid.num_base_patches_j) then
        return i_coord * grid.num_base_patches_j + j_coord
    end
    return -1
end



-- Initialize the meta patches on base level
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
            meta_patches[pid][pid].level   = 0
            meta_patches[pid][pid].i_coord = my_i_coord
            meta_patches[pid][pid].j_coord = my_j_coord
            meta_patches[pid][pid].i_prev  = baseCoordToPid(int(my_i_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid][pid].i_next  = baseCoordToPid(int(my_i_coord + 1                          ) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid][pid].j_prev  = baseCoordToPid(my_i_coord, int(my_j_coord - 1 + grid.num_base_patches_j) % grid.num_base_patches_j)
            meta_patches[pid][pid].j_next  = baseCoordToPid(my_i_coord, int(my_j_coord + 1                          ) % grid.num_base_patches_j)
        else
            meta_patches[pid][pid].level   = -1
            meta_patches[pid][pid].i_coord = -1
            meta_patches[pid][pid].j_coord = -1
            meta_patches[pid][pid].i_prev  = -1
            meta_patches[pid][pid].i_next  = -1
            meta_patches[pid][pid].j_prev  = -1
            meta_patches[pid][pid].j_next  = -1
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
    local __demand(__inline)
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


------ Communication may occur
function grid.fillGhosts(fsp)
    local
    task taskFillGhosts (
        meta_patches             : region(ispace(int1d), grid_meta_fsp),
        meta_patches_part        : partition(disjoint, meta_patches, ispace(int1d)),
        data_patches             : region(ispace(int3d), fsp),
        data_patches_i_prev_send : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_i_next_send : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_j_prev_send : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_j_next_send : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_i_prev_recv : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_i_next_recv : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_j_prev_recv : partition(disjoint, data_patches, ispace(int1d)),
        data_patches_j_next_recv : partition(disjoint, data_patches, ispace(int1d))
    )
    where
        reads (meta_patches.{i_prev, i_next, j_prev, j_next}),
        reads writes (data_patches)
    do
        var csp = ispace(int1d, grid.num_patches_max, 0)
        --__demand(__index_launch)
        for pid in csp do
            var i_prev : int = int(meta_patches_part[pid][pid].i_prev)
            var i_next : int = int(meta_patches_part[pid][pid].i_next)
            var j_prev : int = int(meta_patches_part[pid][pid].j_prev)
            var j_next : int = int(meta_patches_part[pid][pid].j_next)

            if (i_prev > -1) then
                [grid.deepCopy(fsp)](data_patches_i_next_recv[int1d(i_prev)], data_patches_i_prev_send[pid]);
                [grid.deepCopy(fsp)](data_patches_i_prev_recv[pid], data_patches_i_next_send[int1d(i_prev)]);
            end
            
            if (i_next > -1) then
                [grid.deepCopy(fsp)](data_patches_i_prev_recv[int1d(i_next)], data_patches_i_next_send[pid]);
                [grid.deepCopy(fsp)](data_patches_i_next_recv[pid], data_patches_i_prev_send[int1d(i_next)]);
            end

            if (j_prev > -1) then
                [grid.deepCopy(fsp)](data_patches_j_next_recv[int1d(j_prev)], data_patches_j_prev_send[pid]);
                [grid.deepCopy(fsp)](data_patches_j_prev_recv[pid], data_patches_j_next_send[int1d(j_prev)]);
            end
            
            if (j_next > -1) then
                [grid.deepCopy(fsp)](data_patches_j_prev_recv[int1d(j_next)], data_patches_j_next_send[pid]);
                [grid.deepCopy(fsp)](data_patches_j_next_recv[pid], data_patches_j_prev_send[int1d(j_next)]);
            end

        end
    end

    return taskFillGhosts
end



-- Helper task: return the first pid wtih continuous available segment of "length" patches
local task getAvailableSegPatches(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d)),
    length              : int
)
where
    reads writes atomic(meta_patches_region.level)
do
    var offset     : int = grid.num_base_patches_i * grid.num_base_patches_j
    var num_checks : int = grid.num_patches_max - offset - length
    var valid : bool
    for pid in ispace(int1d, num_checks, offset) do
        valid = true
        for j = 0, length-1 do
            var pid_next = pid + int1d(j)
            valid = valid and (meta_patches[pid_next][pid_next].level < 0)
        end
        if (valid) then
            for j = 0, length-1 do
                var pid_next = pid + int1d(j)
                meta_patches[pid_next][pid_next].level = 9999
            end
            return pid
        end
    end
    regentlib.assert(false, "Cannot find any segment of available patches.")
end


-- Enforce the activation of refine requests follows the AMR rule of refinement.
-- TODO: finish this for more complicated refine patterns
local __demand(__inline) task metaRefineFix(
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
            -- TODO: optimize this!!!!!!!!
        end
    end
end



local __demand(__inline) task metaCoarsenFix(
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

            -- TODO: finish this!!!!!!!!
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
task grid.metaRefine(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes atomic(meta_patches_region)
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
            end -- for child_loc
        end
    end -- for pid
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
    var i_prev     : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_prev)
    var i_next     : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].i_next)
    var j_prev     : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].j_prev)
    var j_next     : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].j_next)
    var pid_parent : int1d = int1d(meta_patches[int1d(pid)][int1d(pid)].parent)
    var i_coord    : int   = meta_patches[pid][pid].i_coord
    var j_coord    : int   = meta_patches[pid][pid].j_coord
    var child_loc  : int   = (i_coord % 2) + (j_coord % 2) * 2

    meta_patches[i_prev][i_prev].i_next = -1;
    meta_patches[i_next][i_next].i_prev = -1;
    meta_patches[j_prev][j_prev].j_next = -1;
    meta_patches[j_next][j_next].j_prev = -1;
    meta_patches[pid][pid].level        = -1;
    meta_patches[pid][pid].i_coord      = -1;
    meta_patches[pid][pid].j_coord      = -1;
    meta_patches[pid][pid].parent       = -1;
    meta_patches[pid][pid].refine_req   = false;
    meta_patches[pid][pid].coarsen_req  = false;
    meta_patches[pid][pid].i_prev       = -1;
    meta_patches[pid][pid].j_prev       = -1;
    meta_patches[pid][pid].i_next       = -1;
    meta_patches[pid][pid].j_next       = -1;
    meta_patches[pid_parent][pid_parent].child[child_loc] = -1
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
task grid.metaCoarsen(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads writes (meta_patches_region)
do
    metaCoarsenFix(meta_patches_region, meta_patches)

    for pid in meta_patches.colors do
        var parent_patch = meta_patches[pid][pid];
        if ([_allChildrenAreCoarsenable(parent_patch, meta_patches)]) then
            for child_loc = 0, 4 do
                var child_pid : int1d = parent_patch.child[child_loc];
                _resetLeafMetaPatch(meta_patches_region, meta_patches, child_pid)
            end -- for child_loc
        end
    end -- for pid
end


-- Clear all refine requests after the refinement is completed
__demand(__inline)
task grid.clearRefineReqs(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(meta_patches_region.refine_req)
do
    fill(meta_patches_region.refine_req, false)
end



-- Clear all coarsen requests after the coarsening is completed
__demand(__inline)
task grid.clearCoarsenReqs(
    meta_patches_region : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(meta_patches_region.coarsen_req)
do
    fill(meta_patches_region.coarsen_req, false)
end


return grid
