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

struct grid_meta {
    pid     : int, -- patch id
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



fspace grid_fsp {
    x            : double,
    y            : double,
    refine_flag  : bool,
}


fspace grid_meta_fsp {
    connectivity : grid_meta
}


__demand (__inline)
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

end



__demand (__inline)
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



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



__demand (__inline)
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
    var p = partition(disjoint, complete, patches, coloring, csp)
    c.legion_domain_point_coloring_destroy(coloring)
    return p
end



return grid
