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



task grid.createPartitionOfFullPatches (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfInteriorPatches (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfIPrevSendBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfINextSendBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfIPrevRecvBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfINextRecvBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfJPrevSendBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfJNextSendBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfJPrevRecvBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



task grid.createPartitionOfJNextRecvBuffers (
    patches : region(ispace(int3d), grid_fsp) 
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



terra grid.baseCoordToPid(i_coord : int, j_coord : int) : int
    if (i_coord < grid.num_base_patches_i and j_coord < grid.num_base_patches_j) then
        return i_coord * grid.num_base_patches_j + j_coord
    end
    return -1
end



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
            meta_patches[pid].i_prev  = grid.baseCoordToPid(int(my_i_coord - 1 + grid.num_base_patches_i) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid].i_next  = grid.baseCoordToPid(int(my_i_coord + 1                          ) % grid.num_base_patches_i, my_j_coord)
            meta_patches[pid].j_prev  = grid.baseCoordToPid(my_i_coord, int(my_j_coord - 1 + grid.num_base_patches_j) % grid.num_base_patches_j)
            meta_patches[pid].j_next  = grid.baseCoordToPid(my_i_coord, int(my_j_coord + 1                          ) % grid.num_base_patches_j)
        end
    end
end





return grid
