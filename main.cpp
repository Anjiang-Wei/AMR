#include <stdio.h>
#include "demolibs.h"


//void initBaseGrid(Context& ctx, Runtime* rt, const int fs_size, LogicalRegion& region_of_fields, LogicalPartition& patches_interior, LogicalPartition& patches_extended)
//{
//    constexpr unsigned long NUM_GHOSTS2 = STENCIL_WIDTH - 1;
//    constexpr unsigned long NUM_GHOSTS  = STENCIL_WIDTH / 2;
//    Box2D grid_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X*PATCH_SIZE+NUM_GHOSTS2-1, NUM_PATCHES_Y*PATCH_SIZE+NUM_GHOSTS2-1));
//    IndexSpace grid_isp = rt->create_index_space(ctx, grid_bounds);
//    FieldSpace fs = rt->create_field_space(ctx);
//    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fs);
//    for (int i = 0; i < fs_size; i++) {
//        const int fid = field_allocator.allocate_field(sizeof(Real), i);
//        assert(fid == i);
//    }
//    region_of_fields = rt->create_logical_region(ctx, grid_isp, fs);
//
//    Box2D color_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X+1, NUM_PATCHES_Y+1));
//    IndexSpace color_isp = rt->create_index_space(ctx, color_bounds);
//    std::map<DomainPoint, Domain> domain_map;
//    // Ghost boundaries in x-direction
//    for (auto ip = 0; ip < NUM_PATCHES_X+2; ip+=NUM_PATCHES_X+1) {
//        const unsigned long i_lo = ip>0 ? NUM_GHOSTS + NUM_PATCHES_X * PATCH_SIZE : 0;
//        const unsigned long i_hi = i_lo + NUM_GHOSTS - 1;
//        for (auto jp = 1; jp < NUM_PATCHES_Y+1; jp++) {
//            const unsigned long j_lo = NUM_GHOSTS + (jp-1) * PATCH_SIZE;
//            const unsigned long j_hi = j_lo + PATCH_SIZE - 1;
//            Point2D lower_bounds(i_lo, j_lo);
//            Point2D upper_bounds(i_hi, j_hi);
//            DomainPoint p_coord(Point2D(ip, jp));
//            Domain patch(Box2D(lower_bounds, upper_bounds));
//            domain_map.insert({p_coord, patch});
//            //printf("[DEBUG] x-boundary partition (%2u, %2u): (%2lu, %2lu) -- (%2lu, %2lu)\n", ip, jp, i_lo, j_lo, i_hi, j_hi);
//        }
//    }
//    // Ghost boundaries in y-direction
//    for (auto ip = 1; ip < NUM_PATCHES_X+1; ip++) {
//        const unsigned long i_lo = NUM_GHOSTS + (ip-1) * PATCH_SIZE;
//        const unsigned long i_hi = i_lo + PATCH_SIZE - 1;
//        for (auto jp = 0; jp < NUM_PATCHES_Y+2; jp+=NUM_PATCHES_Y+1) {
//            const unsigned long j_lo = jp>0 ? NUM_GHOSTS + NUM_PATCHES_Y * PATCH_SIZE : 0;
//            const unsigned long j_hi = j_lo + NUM_GHOSTS - 1;
//            Point2D lower_bounds(i_lo, j_lo);
//            Point2D upper_bounds(i_hi, j_hi);
//            DomainPoint p_coord(Point2D(ip, jp));
//            Domain patch(Box2D(lower_bounds, upper_bounds));
//            domain_map.insert({p_coord, patch});
//            //printf("[DEBUG] y-boundary partition (%2u, %2u): (%2lu, %2lu) -- (%2lu, %2lu)\n", ip, jp, i_lo, j_lo, i_hi, j_hi);
//        }
//    }
//
//    // Interior patches
//    for (auto ip = 1; ip < NUM_PATCHES_X+1; ip++) {
//        const unsigned long i_lo = NUM_GHOSTS + (ip-1) * PATCH_SIZE;
//        const unsigned long i_hi = i_lo + PATCH_SIZE - 1;
//        for (auto jp = 1; jp < NUM_PATCHES_Y+1; jp++) {
//            const unsigned long j_lo = NUM_GHOSTS + (jp-1) * PATCH_SIZE;
//            const unsigned long j_hi = j_lo + PATCH_SIZE - 1;
//            Point2D lower_bounds(i_lo, j_lo);
//            Point2D upper_bounds(i_hi, j_hi);
//            DomainPoint p_coord(Point2D(ip, jp));
//            Domain patch(Box2D(lower_bounds, upper_bounds));
//            domain_map.insert({p_coord, patch});
//            //printf("[DEBUG] interior partition (%2u, %2u): (%2lu, %2lu) -- (%2lu, %2lu)\n", ip, jp, i_lo, j_lo, i_hi, j_hi);
//        }
//    }
//    IndexPartition idx_partition = rt->create_partition_by_domain(ctx, grid_isp, domain_map, color_isp);
//    patches_interior = rt->get_logical_partition(ctx, region_of_fields, idx_partition);
//
//    // Transform interior patches
//    Box2D color_bounds_interior = Box2D(Point2D(1,1), Point2D(NUM_PATCHES_X, NUM_PATCHES_Y));
//    IndexSpace color_isp_interior = rt->create_index_space(ctx, color_bounds_interior);
//    Transform2D transform;
//    transform[0][0] = PATCH_SIZE;
//    transform[1][1] = PATCH_SIZE;
//    Box2D extent(Point2D(-PATCH_SIZE, -PATCH_SIZE), Point2D(NUM_GHOSTS2-1, NUM_GHOSTS2-1));
//    IndexPartition idx_part_ext = rt->create_partition_by_restriction(ctx, grid_isp, color_isp_interior, transform, extent);
//    patches_extended = rt->get_logical_partition(ctx, region_of_fields, idx_part_ext);
//}
//
//
////template<typename Op>
////void elem_op_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
////    Op op_functor = task->args;
////}
//
//void stencil_operation_demo_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    Real* args = reinterpret_cast<Real*>(task->args);
//    const Real inv_dx = 1.0 / args[0];
//    const Real inv_dy = 1.0 / args[1];
//    const unsigned long NUM_GHOSTS = STENCIL_WIDTH / 2;
//    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
//    Box2D isp_domain_interior = Box2D(isp_domain.lo+Point2D(NUM_GHOSTS, NUM_GHOSTS), isp_domain.hi-Point2D(NUM_GHOSTS, NUM_GHOSTS));
//    const PhysicalRegion& rgn_pvars = rgns[0];
//    const PhysicalRegion& rgn_bvars = rgns[1];
//    const FieldAccessor<READ_ONLY, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
//    const FieldAccessor<READ_ONLY, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
//    const FieldAccessor<READ_ONLY, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
//    const FieldAccessor<WRITE_DISCARD, Real, 2> dudx(rgn_pvars, static_cast<int>(BVARS_ID::DUDX));
//    const FieldAccessor<WRITE_DISCARD, Real, 2> dudy(rgn_pvars, static_cast<int>(BVARS_ID::DUDY));
//    const FieldAccessor<WRITE_DISCARD, Real, 2> dvdx(rgn_pvars, static_cast<int>(BVARS_ID::DVDX));
//    const FieldAccessor<WRITE_DISCARD, Real, 2> dvdy(rgn_pvars, static_cast<int>(BVARS_ID::DVDY));
//    for (PointInBox2D ij(isp_domain_interior); ij(); ij++) {
//            Point2D ij_e1 = *ij + Point2D( 1, 0);
//            Point2D ij_w1 = *ij + Point2D(-1, 0);
//            Point2D ij_n1 = *ij + Point2D( 0,+1);
//            Point2D ij_s1 = *ij + Point2D( 0,-1);
//            dudx[*ij] = 0.5 * inv_dx * (u[ij_e1] - u[ij_w1]);
//            dvdx[*ij] = 0.5 * inv_dx * (v[ij_e1] - v[ij_w1]);
//            dudy[*ij] = 0.5 * inv_dy * (u[ij_n1] - u[ij_s1]);
//            dvdy[*ij] = 0.5 * inv_dy * (v[ij_n1] - v[ij_s1]);
//    }
//}
//
//void set_initial_condition_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//
//    printf("Launching \"set_initial_condition_task\":\n");
//    const PhysicalRegion& rgn_cvars = rgns[0];
//    const PhysicalRegion& rgn_pvars = rgns[1];
//    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
//
//    // Set primitive variables
//    {
//        const FieldAccessor<WRITE_DISCARD, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
//        const FieldAccessor<WRITE_DISCARD, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
//        const FieldAccessor<WRITE_DISCARD, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
//        const FieldAccessor<WRITE_DISCARD, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
//        for (PointInBox2D ij(isp_domain); ij(); ij++) {
//            rho[*ij] = 1.0;
//            u  [*ij] = 0.0;
//            v  [*ij] = 0.0;
//            T  [*ij] = 1.0;
//        }
//    }
//
//    // Convert primitive variables to conservative variables
//    {
//        const FieldAccessor<READ_ONLY, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
//        const FieldAccessor<READ_ONLY, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
//        const FieldAccessor<READ_ONLY, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
//        const FieldAccessor<READ_ONLY, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
//        const FieldAccessor<WRITE_DISCARD, Real, 2> mass (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MASS_0));
//        const FieldAccessor<WRITE_DISCARD, Real, 2> mmtx (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTX_0));
//        const FieldAccessor<WRITE_DISCARD, Real, 2> mmty (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTY_0));
//        const FieldAccessor<WRITE_DISCARD, Real, 2> enrg (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::ENRG_0));
//        for (PointInBox2D ij(isp_domain); ij(); ij++) {
//            mass[*ij] = rho[*ij];
//            mmtx[*ij] =   u[*ij];
//            mmty[*ij] =   v[*ij];
//            enrg[*ij] =   T[*ij];
//        }
//    }
//}
//
//
//void calc_rhs_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    constexpr int OVARS_RGN_ID = 0;
//    constexpr int CVARS_RGN_ID = 1;
//    constexpr int PVARS_RGN_ID = 2;
//    constexpr int BVARS_RGN_ID = 3;
//
//    // TODO: for now only set zeros
//    Real* args = reinterpret_cast<Real*>(task->args);
//    const Real arg0 = args[0];
//
//}
//
//
//
//void ssp_rk3_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    // TODO: Launch calc_rhs_task 3 times but using different stages of CVARS
//}
//
//void top_level_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    int argc    = Legion::Runtime::get_input_args().argc;
//    char** argv = Legion::Runtime::get_input_args().argv;
//
//    LogicalRegion    rgn_coord;
//    LogicalRegion    rgn_cvars;
//    LogicalRegion    rgn_pvars;
//    LogicalRegion    rgn_bvars;
//    LogicalPartition patches_int_coord;
//    LogicalPartition patches_int_cvars;
//    LogicalPartition patches_int_pvars;
//    LogicalPartition patches_int_bvars;
//    LogicalPartition patches_ext_coord;
//    LogicalPartition patches_ext_cvars;
//    LogicalPartition patches_ext_pvars;
//    LogicalPartition patches_ext_bvars;
//    Box2D color_bounds_interior = Box2D(Point2D(1,1), Point2D(NUM_PATCHES_X, NUM_PATCHES_Y));
//    IndexSpace color_isp_interior = rt->create_index_space(ctx, color_bounds_interior);
//
//    printf("Call \"initBaseGrid\" from \"top_level_task\" ... ");
//    initBaseGrid(ctx, rt, static_cast<int>(STAGE3_CVARS_ID::SIZE), rgn_cvars, patches_int_cvars, patches_ext_cvars);
//    initBaseGrid(ctx, rt, static_cast<int>(       PVARS_ID::SIZE), rgn_pvars, patches_int_pvars, patches_ext_pvars);
//    initBaseGrid(ctx, rt, static_cast<int>(       BVARS_ID::SIZE), rgn_bvars, patches_int_bvars, patches_ext_bvars);
//    initBaseGrid(ctx, rt, static_cast<int>(       COORD_ID::SIZE), rgn_coord, patches_int_coord, patches_ext_coord);
//    printf("Done!\n");
//
//    printf("\n\nSet initial conditions ... ");
//    IndexLauncher index_launcher(static_cast<int>(TASK_ID::SET_INIT_COND), color_isp_interior, TaskArgument(NULL, 0), ArgumentMap());
//    index_launcher.add_region_requirement(RegionRequirement(patches_int_cvars, 0, WRITE_DISCARD, EXCLUSIVE, rgn_cvars));
//    index_launcher.add_region_requirement(RegionRequirement(patches_int_pvars, 0,    READ_WRITE, EXCLUSIVE, rgn_pvars));
//    index_launcher.region_requirements[0].add_field(static_cast<int>(STAGE3_CVARS_ID::MASS_0));
//    index_launcher.region_requirements[0].add_field(static_cast<int>(STAGE3_CVARS_ID::MMTX_0));
//    index_launcher.region_requirements[0].add_field(static_cast<int>(STAGE3_CVARS_ID::MMTY_0));
//    index_launcher.region_requirements[0].add_field(static_cast<int>(STAGE3_CVARS_ID::ENRG_0));
//    index_launcher.region_requirements[1].add_field(static_cast<int>(PVARS_ID::DENSITY));
//    index_launcher.region_requirements[1].add_field(static_cast<int>(PVARS_ID::VEL_X)  );
//    index_launcher.region_requirements[1].add_field(static_cast<int>(PVARS_ID::VEL_Y)  );
//    index_launcher.region_requirements[1].add_field(static_cast<int>(PVARS_ID::TEMP)   );
//    rt->execute_index_space(ctx, index_launcher);
//    
//    printf("Done!\n");
//
//    printf("\n\nStart time integration:\n");
//    printf("Simulation done!!!\n");
//
//#if (USE_HDF5)
//    // Prepare for copy
//    FieldSpace cp_fs = rt->create_field_space(ctx);
//    {
//        FieldAllocator allocator = rt->create_field_allocator(ctx, cp_fs);
//        allocator.allocate_field(sizeof(Real), static_cast<int>(COPY_ID::FID_CP));
//    }
//    LogicalRegion cp_lr = rt->create_logical_region(ctx, rgn_coord.get_index_space(), cp_fs);
//
//
//
//    char hdf5_file_name[256];
//    char hdf5_dataset_name[256];
//    strcpy(hdf5_file_name, "filename");
//    strcpy(hdf5_dataset_name, "FID_CP");
//
//    PhysicalRegion cp_pr;
//
//    // create the HDF5 file first - attach wants it to already exist
//    int num_elements = 1000;
//    bool ok = generate_hdf_file(hdf5_file_name, hdf5_dataset_name, num_elements);
//    assert(ok);
//    std::map<FieldID,const char*> field_map;
//    field_map[static_cast<Legion::FieldID>(PVARS_ID::VEL_X)] = hdf5_dataset_name;
//    printf("Checkpointing data to HDF5 file '%s' (dataset='%s')\n",
//            hdf5_file_name, hdf5_dataset_name);
//    AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, rgn_cvars, rgn_cvars);
//    al.attach_hdf5(hdf5_file_name, field_map, LEGION_FILE_READ_WRITE);
//    cp_pr = rt->attach_external_resource(ctx, al);
//
//    CopyLauncher copy_launcher;
//    copy_launcher.add_copy_requirements(
//        RegionRequirement(rgn_coord, READ_ONLY, EXCLUSIVE, rgn_coord),
//        RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
//    copy_launcher.add_src_field(0, static_cast<Legion::FieldID>(PVARS_ID::VEL_X));
//    copy_launcher.add_dst_field(0, static_cast<Legion::FieldID>(COPY_ID::FID_CP));
//    rt->issue_copy_operation(ctx, copy_launcher);
//
//    Future f = rt->detach_external_resource(ctx, cp_pr);
//    f.get_void_result(true /*silence warnings*/);
//#endif
//}
//



int main(int argc, char* argv[]) {

    printf("Hello World from the real main test!\n");
    registerAllTasks();
    return Legion::Runtime::start(argc, argv);

}



