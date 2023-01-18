#include <stdio.h>
#include "util.h"

constexpr unsigned long PATCH_SIZE = 8;
constexpr unsigned long NUM_PATCHES_X = 7;
constexpr unsigned long NUM_PATCHES_Y = 7;

enum FSPACE_ID {
    VEL_X, VEL_Y, VEL_Z, DENSITY, TEMP
};


void initBaseGrid(Context& ctx, Runtime* rt, LogicalRegion& region_of_fields, LogicalPartition& patches)
{
    Box2D grid_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X*PATCH_SIZE-1, NUM_PATCHES_Y*PATCH_SIZE-1));
    IndexSpace grid_isp = rt->create_index_space(ctx, grid_bounds);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fs);
    FieldID fid_vel_x = field_allocator.allocate_field(sizeof(Real), FSPACE_ID::VEL_X);
    FieldID fid_vel_y = field_allocator.allocate_field(sizeof(Real), FSPACE_ID::VEL_Y);
    FieldID fid_vel_z = field_allocator.allocate_field(sizeof(Real), FSPACE_ID::VEL_Z);
    FieldID fid_rho   = field_allocator.allocate_field(sizeof(Real), FSPACE_ID::DENSITY);
    FieldID fid_T     = field_allocator.allocate_field(sizeof(Real), FSPACE_ID::TEMP);
    region_of_fields = rt->create_logical_region(ctx, grid_isp, fs);

    Box2D color_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X, NUM_PATCHES_Y));
    IndexSpace color_isp = rt->create_index_space(ctx, color_bounds);
    std::map<DomainPoint, Domain> domain_map;
    for (auto ip = 0; ip < NUM_PATCHES_X; ip++) {
        const unsigned long i_lo = ip * PATCH_SIZE;
        const unsigned long i_hi = i_lo + PATCH_SIZE - 1;
        for (auto jp = 0; jp < NUM_PATCHES_Y; jp++) {
            const unsigned long j_lo = jp * PATCH_SIZE;
            const unsigned long j_hi = j_lo + PATCH_SIZE - 1;
            Point2D lower_bounds(i_lo, j_lo);
            Point2D upper_bounds(i_hi, j_hi);
            DomainPoint p_coord(Point2D(ip, jp));
            Domain patch(Box2D(lower_bounds, upper_bounds));
            domain_map.insert({p_coord, patch});
        }
    }
    IndexPartition idx_partition = rt->create_partition_by_domain(ctx, grid_isp, domain_map, color_isp);
    patches = rt->get_logical_partition(ctx, region_of_fields, idx_partition);
}



void top_level_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;

    LogicalRegion region_of_fields;
    LogicalPartition patches;

    printf("Call \"initBaseGrid\" from \"top_level_task\" ... ");
    initBaseGrid(ctx, rt, region_of_fields, patches);
    printf("Done!\n");

}




int main(int argc, char* argv[]) {

    printf("Hello World from the real main test!\n");

    Legion::Runtime::set_top_level_task_id(MAIN_TASK_ID);

    {
        Legion::TaskVariantRegistrar registrar(MAIN_TASK_ID, "top_level_task");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
    }

    return Legion::Runtime::start(argc, argv);

}



