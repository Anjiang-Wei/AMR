
#include "util.h"

#if(0)
IndexSpace getColorIndexSpaceInt(Context& ctx, Runtime* rt, const BaseGridConfig& grid_config) {
    Box2D color_bounds_int = Box2D(Point2D(1,1), Point2D(grid_config.NUM_PATCHES_X, grid_config.NUM_PATCHES_Y));
    return rt->create_index_space(ctx, color_bounds_int);
}



IndexSpace getColorIndexSpaceExt(Context& ctx, Runtime* rt, const BaseGridConfig& grid_config) {
    Box2D color_bounds_ext = Box2D(Point2D(0,0), Point2D(grid_config.NUM_PATCHES_X+1, grid_config.NUM_PATCHES_Y+1));
    return rt->create_index_space(ctx, color_bounds_ext);
}




void initializeBaseGrid2D(Context& ctx, Runtime* rt, const BaseGridConfig grid_config, const uint_t num_fields, RegionOfFields& region_of_fields) {
    const uint_t PATCH_SIZE    = grid_config.PATCH_SIZE;
    const uint_t NUM_PATCHES_X = grid_config.NUM_PATCHES_X;
    const uint_t NUM_PATCHES_Y = grid_config.NUM_PATCHES_Y;
    const uint_t STENCIL_WIDTH = grid_config.STENCIL_WIDTH * 2 - 1;

    const uint_t NUM_GHOSTS            = STENCIL_WIDTH / 2;
    const uint_t NUM_GRID_POINTS_X_EXT = NUM_GHOSTS * 2 + NUM_PATCHES_X * PATCH_SIZE;
    const uint_t NUM_GRID_POINTS_Y_EXT = NUM_GHOSTS * 2 + NUM_PATCHES_Y * PATCH_SIZE;

    Box2D      grid_bounds_ext  = Box2D(Point2D(0,0), Point2D(NUM_GRID_POINTS_X_EXT-1, NUM_GRID_POINTS_Y_EXT-1));
    IndexSpace grid_isp_ext     = rt->create_index_space(ctx, grid_bounds_ext);
    FieldSpace fspace           = rt->create_field_space(ctx);
    IndexSpace color_isp_ext    = getColorIndexSpaceExt(ctx, rt, grid_config);
    IndexSpace color_isp_int    = getColorIndexSpaceInt(ctx, rt, grid_config);


    /*** ALLOCATE FIELDS ***/
    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fspace);
    for (uint_t i = 0; i < num_fields; i++) {
        uint_t fid = field_allocator.allocate_field(sizeof(Real), i);
        assert(fid == i);
    }
    region_of_fields.region = rt->create_logical_region(ctx, grid_isp_ext, fspace);


    /*** CREATE LOGICAL PARTITIONS ***/
    // Ghost boundary partitions in x-direction
    std::map<DomainPoint, Domain> domain_map_ghost_x;
    for (uint_t ip = 0; ip < NUM_PATCHES_X+2; ip+=NUM_PATCHES_X+1) {
        const uint_t i_lo = ip>0 ? NUM_GHOSTS + NUM_PATCHES_X * PATCH_SIZE : 0;
        const uint_t i_hi = i_lo + NUM_GHOSTS - 1;
        for (uint_t jp = 1; jp < NUM_PATCHES_Y+1; jp++) {
            const uint_t j_lo = NUM_GHOSTS + (jp-1) * PATCH_SIZE;
            const uint_t j_hi = j_lo + PATCH_SIZE - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map_ghost_x.insert({p_coord, patch});
        }
    }
    IndexPartition  idx_part_ghost_x = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_x, color_isp_ext);
    region_of_fields.patches_ghost_x = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_x);

    // Ghost boundaries in y-direction
    std::map<DomainPoint, Domain> domain_map_ghost_y;
    for (uint_t ip = 1; ip < NUM_PATCHES_X+1; ip++) {
        const uint_t i_lo = NUM_GHOSTS + (ip-1) * PATCH_SIZE;
        const uint_t i_hi = i_lo + PATCH_SIZE - 1;
        for (uint_t jp = 0; jp < NUM_PATCHES_Y+2; jp+=NUM_PATCHES_Y+1) {
            const uint_t j_lo = jp>0 ? NUM_GHOSTS + NUM_PATCHES_Y * PATCH_SIZE : 0;
            const uint_t j_hi = j_lo + NUM_GHOSTS - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map_ghost_y.insert({p_coord, patch});
        }
    }
    IndexPartition  idx_part_ghost_y = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_y, color_isp_ext);
    region_of_fields.patches_ghost_y = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_y);

    // Interior patches
    std::map<DomainPoint, Domain> domain_map_int;
    for (uint_t ip = 1; ip < NUM_PATCHES_X+1; ip++) {
        const uint_t i_lo = NUM_GHOSTS + (ip-1) * PATCH_SIZE;
        const uint_t i_hi = i_lo + PATCH_SIZE - 1;
        for (uint_t jp = 1; jp < NUM_PATCHES_Y+1; jp++) {
            const uint_t j_lo = NUM_GHOSTS + (jp-1) * PATCH_SIZE;
            const uint_t j_hi = j_lo + PATCH_SIZE - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map_int.insert({p_coord, patch});
        }
    }
    IndexPartition idx_part_int = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_int, color_isp_ext);
    region_of_fields.patches_int = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_int);


    /*** TRANSFORM INTERIOR PATCHES ***/
    Transform2D transform;
    transform[0][0] = PATCH_SIZE;
    transform[1][1] = PATCH_SIZE;
    Box2D extent(Point2D(-PATCH_SIZE, -PATCH_SIZE), Point2D(NUM_GHOSTS*2-1, NUM_GHOSTS*2-1));
    IndexPartition idx_part_ext = rt->create_partition_by_restriction(ctx, grid_isp_ext, color_isp_int, transform, extent);
    region_of_fields.patches_ext = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ext);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////


void initializeBaseGrid2DNew(Context& ctx, Runtime* rt, const BaseGridConfig grid_config, const uint_t num_fields, RegionOfFields& region_of_fields) {
    const uint_t PATCH_SIZE    = grid_config.PATCH_SIZE;
    const uint_t NUM_PATCHES_X = grid_config.NUM_PATCHES_X;
    const uint_t NUM_PATCHES_Y = grid_config.NUM_PATCHES_Y;
    const uint_t STENCIL_WIDTH = grid_config.STENCIL_WIDTH * 2 - 3; // consider the convective and diffusive operators

    const int_t NUM_GHOSTS            = STENCIL_WIDTH / 2;
    const int_t NUM_GRID_POINTS_X_INT = NUM_PATCHES_X * PATCH_SIZE;
    const int_t NUM_GRID_POINTS_Y_INT = NUM_PATCHES_Y * PATCH_SIZE;
    const int_t NUM_GRID_POINTS_X_EXT = NUM_GHOSTS * 2 + NUM_PATCHES_X * PATCH_SIZE;
    const int_t NUM_GRID_POINTS_Y_EXT = NUM_GHOSTS * 2 + NUM_PATCHES_Y * PATCH_SIZE;

    /*** ALLOCATE FIELDS ***/
    FieldSpace fspace = rt->create_field_space(ctx);
    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fspace);
    for (uint_t i = 0; i < num_fields; i++) {
        uint_t fid = field_allocator.allocate_field(sizeof(Real), i);
        assert(fid == i);
    }


    /*** CREATE GLOBAL GRID AND COLOR INDEX SPACES ***/
    //Box2D      grid_bounds_int = Box2D(Point2D(0, 0), Point2D(NUM_GRID_POINTS_X_INT - 1, NUM_GRID_POINTS_X_INT - 1));
    Box2D      grid_bounds_ext = Box2D(Point2D(-NUM_GHOSTS, -NUM_GHOSTS), Point2D(NUM_GHOSTS + NUM_GRID_POINTS_X_INT - 1, NUM_GHOSTS + NUM_GRID_POINTS_Y_INT - 1));
    Box2D     color_bounds_int = Box2D(Point2D(0,0), Point2D(grid_config.NUM_PATCHES_X-1, grid_config.NUM_PATCHES_Y-1));
    Box2D color_bounds_ghost_x = Box2D(Point2D(0,0), Point2D(0, grid_config.NUM_PATCHES_Y-1));
    Box2D color_bounds_ghost_y = Box2D(Point2D(0,0), Point2D(grid_config.NUM_PATCHES_X-1, 0));
    //IndexSpace grid_isp_int      = rt->create_index_space(ctx, grid_bounds_int);
    IndexSpace grid_isp_ext      = rt->create_index_space(ctx, grid_bounds_ext);
    IndexSpace color_isp_int     = rt->create_index_space(ctx, color_bounds_int);
    IndexSpace color_isp_ghost_x = rt->create_index_space(ctx, color_bounds_ghost_x);
    IndexSpace color_isp_ghost_y = rt->create_index_space(ctx, color_bounds_ghost_y);
    IndexSpace grid_isp_ghost_x_lo = rt->create_index_space(ctx, Box2D(Point2D(-NUM_GHOSTS, 0), Point2D(-1, NUM_GRID_POINTS_Y_INT - 1)));
    IndexSpace grid_isp_ghost_x_hi = rt->create_index_space(ctx, Box2D(Point2D(NUM_GRID_POINTS_X_INT, 0), Point2D(NUM_GRID_POINTS_X_INT + NUM_GHOSTS - 1, NUM_GRID_POINTS_Y_INT - 1)));
    IndexSpace grid_isp_ghost_mirror_x_lo = rt->create_index_space(ctx, Box2D(Point2D(0, 0), Point2D(NUM_GHOSTS - 1, NUM_GRID_POINTS_Y_INT - 1)));
    IndexSpace grid_isp_ghost_mirror_x_hi = rt->create_index_space(ctx, Box2D(Point2D(NUM_GRID_POINTS_X_INT - NUM_GHOSTS, 0), Point2D(NUM_GRID_POINTS_X_INT - 1, NUM_GRID_POINTS_Y_INT - 1)));
    IndexSpace grid_isp_ghost_y_lo = rt->create_index_space(ctx, Box2D(Point2D(0, -NUM_GHOSTS), Point2D(NUM_GRID_POINTS_X_INT - 1, -1)));
    IndexSpace grid_isp_ghost_y_hi = rt->create_index_space(ctx, Box2D(Point2D(0, NUM_GRID_POINTS_Y_INT), Point2D(NUM_GRID_POINTS_X_INT - 1, NUM_GRID_POINTS_Y_INT + NUM_GHOSTS - 1)));
    IndexSpace grid_isp_ghost_mirror_y_lo = rt->create_index_space(ctx, Box2D(Point2D(0, 0), Point2D(NUM_GRID_POINTS_X_INT - 1, NUM_GHOSTS - 1)));
    IndexSpace grid_isp_ghost_mirror_y_hi = rt->create_index_space(ctx, Box2D(Point2D(0, NUM_GRID_POINTS_Y_INT - NUM_GHOSTS), Point2D(NUM_GRID_POINTS_X_INT - 1, NUM_GRID_POINTS_Y_INT - 1)));

    /*** CREATE LOGICAL PARTITIONS ***/
    // Create parent region
    region_of_fields.region = rt->create_logical_region(ctx, grid_isp_ext, fspace);

    // Ghost boundary partitions in x-direction
    std::map<DomainPoint, Domain> domain_map_ghost_x_lo;
    std::map<DomainPoint, Domain> domain_map_ghost_x_hi;
    std::map<DomainPoint, Domain> domain_map_ghost_mirror_x_lo;
    std::map<DomainPoint, Domain> domain_map_ghost_mirror_x_hi;
    for (uint_t jp = 0; jp < NUM_PATCHES_Y; jp++) {
        const uint_t j_lo = jp * PATCH_SIZE;
        const uint_t j_hi = j_lo + PATCH_SIZE - 1;
        DomainPoint p_coord( Point2D(0, jp) );
        Domain domain_ghost_x_lo( Box2D(Point2D(-NUM_GHOSTS, j_lo), Point2D(-1, j_hi)) );
        Domain domain_ghost_x_hi( Box2D(Point2D(NUM_GRID_POINTS_X_INT, j_lo), Point2D(NUM_GRID_POINTS_X_INT + NUM_GHOSTS - 1, j_hi)) );
        Domain domain_ghost_mirror_x_lo( Box2D(Point2D(0, j_lo), Point2D(NUM_GHOSTS - 1, j_hi)) );
        Domain domain_ghost_mirror_x_hi( Box2D(Point2D(NUM_GRID_POINTS_X_INT - NUM_GHOSTS, j_lo), Point2D(NUM_GRID_POINTS_X_INT - 1, j_hi)) );
        domain_map_ghost_x_lo.insert({p_coord, domain_ghost_x_lo});
        domain_map_ghost_x_hi.insert({p_coord, domain_ghost_x_hi});
        domain_map_ghost_mirror_x_lo.insert({p_coord, domain_ghost_mirror_x_lo});
        domain_map_ghost_mirror_x_hi.insert({p_coord, domain_ghost_mirror_x_hi});
    }
    IndexPartition  idx_part_ghost_x_lo        = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_x_lo,        color_isp_ghost_x);
    IndexPartition  idx_part_ghost_x_hi        = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_x_hi,        color_isp_ghost_x);
    IndexPartition  idx_part_ghost_mirror_x_lo = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_mirror_x_lo, color_isp_ghost_x);
    IndexPartition  idx_part_ghost_mirror_x_hi = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_mirror_x_hi, color_isp_ghost_x);
    region_of_fields.patches_ghost_x_lo        = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_x_lo);
    region_of_fields.patches_ghost_x_hi        = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_x_hi);
    region_of_fields.patches_ghost_mirror_x_lo = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_mirror_x_lo);
    region_of_fields.patches_ghost_mirror_x_hi = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_mirror_x_hi);

    // Ghost boundary partitions in y-direction
    std::map<DomainPoint, Domain> domain_map_ghost_y_lo;
    std::map<DomainPoint, Domain> domain_map_ghost_y_hi;
    std::map<DomainPoint, Domain> domain_map_ghost_mirror_y_lo;
    std::map<DomainPoint, Domain> domain_map_ghost_mirror_y_hi;
    for (uint_t ip = 0; ip < NUM_PATCHES_X; ip++) {
        const uint_t i_lo = ip * PATCH_SIZE;
        const uint_t i_hi = i_lo + PATCH_SIZE - 1;
        DomainPoint p_coord( Point2D(ip, 0) );
        Domain domain_ghost_y_lo( Box2D(Point2D(i_lo, -NUM_GHOSTS), Point2D(i_hi, -1)) );
        Domain domain_ghost_y_hi( Box2D(Point2D(i_lo, NUM_GRID_POINTS_Y_INT), Point2D(i_hi, NUM_GRID_POINTS_Y_INT + NUM_GHOSTS - 1)) );
        Domain domain_ghost_mirror_y_lo( Box2D(Point2D(i_lo, 0), Point2D(i_hi, NUM_GHOSTS - 1)) );
        Domain domain_ghost_mirror_y_hi( Box2D(Point2D(i_lo, NUM_GRID_POINTS_Y_INT - NUM_GHOSTS), Point2D(i_hi, NUM_GRID_POINTS_Y_INT - 1)) );
        domain_map_ghost_y_lo.insert({p_coord, domain_ghost_y_lo});
        domain_map_ghost_y_hi.insert({p_coord, domain_ghost_y_hi});
        domain_map_ghost_mirror_y_lo.insert({p_coord, domain_ghost_mirror_y_lo});
        domain_map_ghost_mirror_y_hi.insert({p_coord, domain_ghost_mirror_y_hi});
    }
    IndexPartition  idx_part_ghost_y_lo        = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_y_lo,        color_isp_ghost_y);
    IndexPartition  idx_part_ghost_y_hi        = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_y_hi,        color_isp_ghost_y);
    IndexPartition  idx_part_ghost_mirror_y_lo = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_mirror_y_lo, color_isp_ghost_y);
    IndexPartition  idx_part_ghost_mirror_y_hi = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_ghost_mirror_y_hi, color_isp_ghost_y);
    region_of_fields.patches_ghost_y_lo        = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_y_lo);
    region_of_fields.patches_ghost_y_hi        = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_y_hi);
    region_of_fields.patches_ghost_mirror_y_lo = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_mirror_y_lo);
    region_of_fields.patches_ghost_mirror_y_hi = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ghost_mirror_y_hi);

    // Interior patches
    std::map<DomainPoint, Domain> domain_map_int;
    for (uint_t ip = 0; ip < NUM_PATCHES_X; ip++) {
        const uint_t i_lo = ip * PATCH_SIZE;
        const uint_t i_hi = i_lo + PATCH_SIZE - 1;
        for (uint_t jp = 0; jp < NUM_PATCHES_Y; jp++) {
            const uint_t j_lo = jp * PATCH_SIZE;
            const uint_t j_hi = j_lo + PATCH_SIZE - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map_int.insert({p_coord, patch});
        }
    }
    IndexPartition idx_part_int = rt->create_partition_by_domain(ctx, grid_isp_ext, domain_map_int, color_isp_int);
    region_of_fields.patches_int = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_int);

    /*** TRANSFORM INTERIOR PATCHES ***/
    Transform2D transform;
    transform[0][0] = PATCH_SIZE;
    transform[1][1] = PATCH_SIZE;
    Box2D extent(Point2D(-NUM_GHOSTS, -NUM_GHOSTS), Point2D(PATCH_SIZE + NUM_GHOSTS - 1, PATCH_SIZE + NUM_GHOSTS - 1));
    IndexPartition idx_part_ext = rt->create_partition_by_restriction(ctx, grid_isp_ext, color_isp_int, transform, extent);
    region_of_fields.patches_ext = rt->get_logical_partition(ctx, region_of_fields.region, idx_part_ext);
}



