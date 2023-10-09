import "regent"
local c = regentlib.c


local grid = {}


if grid.num_base_patches_i == nil then grid.num_base_patches_i = 4 end
if grid.num_base_patches_j == nil then grid.num_base_patches_j = 4 end
if grid.patch_size         == nil then grid.patch_size = 16 end
if grid.level_max          == nil then grid.level_max = 1 end
if grid.num_ghosts         == nil then grid.num_ghosts = 4 end
if grid.num_patches_max    == nil then grid.num_patches_max = (bit.lshift(1, 2*grid.level_max)) * grid.num_base_patches_i * grid.num_base_patches_j end

grid.full_patch_size = grid.patch_size + 2 * grid.num_ghosts -- patch size including interior region and ghost region
grid.idx_min         = -grid.num_ghosts                      -- starting index of the patch index space in each dimension including ghost region
grid.idx_max         = grid.patch_size - 1 + grid.num_ghosts -- last index of the patch index space in each dimension including ghost region



fspace grid_fsp {
    x            : double,
    y            : double,
    refine_flag  : bool,
}


fspace grid_meta_fsp {
    level   : int, -- level of grid
    i_coord : int, -- patch coordinate in i-dimension
    j_coord : int, -- patch coordinate in j-dimension
    i_prev  : int, -- patch id of the neighboring patch ahead in i-dimension
    i_next  : int, -- patch id of the neighboring patch after in i-dimension
    j_prev  : int, -- patch id of the neighboring patch ahead in j-dimension
    j_next  : int, -- patch id of the neighboring patch after in j-dimension
    parent  : int, -- patch id of the parent patch
    child1  : int, -- patch id of the 1st child patch
    child2  : int, -- patch id of the 2nd child patch
    child3  : int, -- patch id of the 3rd child patch
    child4  : int  -- patch id of the 4th child patch
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
task grid.baseMetaGridInit(
    meta_patches : region(ispace(int1d), grid_meta_fsp)
)
where
    writes (meta_patches.{level, i_coord, j_coord, i_next, j_next, i_prev, j_prev, parent, child1, child2, child3, child4})
do
    var num_base_patches : int = grid.num_base_patches_i * grid.num_base_patches_j
    for pid in meta_patches.ispace do
        if (int(pid) < num_base_patches) then
            var my_i_coord : int = int(pid) / grid.num_base_patches_j
            var my_j_coord : int = int(pid) % grid.num_base_patches_j
            meta_patches[pid].level   = 0
            meta_patches[pid].i_coord = my_i_coord
            meta_patches[pid].j_coord = my_j_coord
            meta_patches[pid].i_prev  = baseCoordToPid(int(my_i_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid].i_next  = baseCoordToPid(int(my_i_coord + 1                          ) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid].j_prev  = baseCoordToPid(my_i_coord, int(my_j_coord - 1 + grid.num_base_patches_j) % grid.num_base_patches_j)
            meta_patches[pid].j_next  = baseCoordToPid(my_i_coord, int(my_j_coord + 1                          ) % grid.num_base_patches_j)
        end
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



-- Get i_prev of one meta-patch
-- meta_patch is a sub-region of meta_patches
task grid.getIPrev(meta_patch : region(ispace(int1d), grid_meta_fsp))
where
    reads(meta_patch.i_prev)
do
    regentlib.assert(meta_patch.ispace.bounds.hi == meta_patch.ispace.bounds.lo, "Sub-region with only one patch must be passed to the task.")
    var pid : int1d = meta_patch.ispace.bounds.lo
    var i_prev : int = int(meta_patch[pid].i_prev)
    return i_prev
end


function grid.fillGhosts(part_patches_meta, part_patches_grid, fsp)
    return rexpr var color_space = [grid.createColorSpace()] end -- rexpr
end


---- Fill ghosts points (recv buffers) for all fields in one region
---- Communication may occur
--function grid.fillGhosts(part_patches_meta, part_patches_grid, fsp)
--    local __demand(__inline) -- require index launch
--    task taskGetMetaDataIPrev(
--        meta_patches : region(ispace(int1d), grid_meta_fsp)
--    )
--    where
--        reads(meta_patches.i_prev)
--    do
--        regentlib.assert(meta_patches.ispace.bounds.hi == meta_patches.ispace.bounds.lo)
--        var pid : int1d = meta_patches.ispace.bounds.lo
--        var i_prev : int = int(meta_patches[pid].i_prev)
--        return i_prev
--    end
--
--    color_space = grid.createColorSpace()
--
--    __demand(__index_launch)
--    for pid in color_space do
--        var i_prev : int = taskGetMetaDataIPrev(part_patches_meta[pid])
--        if (i_prev > 0) then
--            [grid.deepCopy(fsp)](part_patches_grid[pid], part_patches_grid[i_prev])
--        end
--    end

    --local __demand(__inline) -- require index launch
    --task taskGetMetaDataJPrev(
    --    meta_patches : region(ispace(int1d), grid_meta_fsp)
    --)
    --where
    --    reads(meta_patches.j_prev)
    --do
    --    regentlib.assert(meta_patches.ispace.bounds.hi == meta_patches.ispace.bounds.lo)
    --    var pid : int1d = meta_patches.ispace.bounds.lo
    --    var j_prev : int = int(meta_patches[pid].j_prev)
    --    return j_prev
    --end

    --local __demand(__inline) -- require index launch
    --task taskGetMetaDataINext(
    --    meta_patches : region(ispace(int1d), grid_meta_fsp)
    --)
    --where
    --    reads(meta_patches.i_next)
    --do
    --    regentlib.assert(meta_patches.ispace.bounds.hi == meta_patches.ispace.bounds.lo)
    --    var pid : int1d = meta_patches.ispace.bounds.lo
    --    var i_next : int = int(meta_patches[pid].i_next)
    --    return i_next
    --end

    --local __demand(__inline) -- require index launch
    --task taskGetMetaDataJNext(
    --    meta_patches : region(ispace(int1d), grid_meta_fsp)
    --)
    --where
    --    reads(meta_patches.j_next)
    --do
    --    regentlib.assert(meta_patches.ispace.bounds.hi == meta_patches.ispace.bounds.lo)
    --    var pid : int1d = meta_patches.ispace.bounds.lo
    --    var j_next : int = int(meta_patches[pid].j_next)
    --    return j_next
    --end

    --int1d color_space = [grid.createColorSpace()] -- ispace(int1d, grid.num_patches_max, 0)

--end

return grid
