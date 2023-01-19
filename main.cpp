#include <stdio.h>
#include "util.h"
#include <hdf5.h>
#include "numerics.h"

constexpr unsigned long PATCH_SIZE    = 8;
constexpr unsigned long NUM_PATCHES_X = 8;
constexpr unsigned long NUM_PATCHES_Y = 8;
constexpr Real          t_final       = 10.0;
constexpr Real          viscosity     = 10.0;
constexpr Real          Prandtl       = 1.0;
constexpr Real          Rg            = 1.0;
constexpr Real          gamma         = 1.4;

// Fieldspace of primitive variables
enum class PVARS_ID {
    VEL_X, VEL_Y, DENSITY, TEMP,
    SIZE
};


// Fieldspace of conservative variables
enum class STAGE3_CVARS_ID {
   MASS_0, MMTX_0, MMTY_0, ENRG_0, 
   MASS_1, MMTX_1, MMTY_1, ENRG_1, 
   MASS_2, MMTX_2, MMTY_2, ENRG_2, 
   SIZE
};

// Fieldspace of buffer variables
enum class BVARS_ID {
    PRSES,
    SIZE
};


// Mesh
enum class COORD_ID {
    X, Y, Z, SIZE
};

enum class COPY_ID {
    FID_CP,
};


enum TASK_ID {
    TOP_LEVEL, ELEM_OP
};


bool generate_hdf_file(const char *file_name, const char *dataset_name, int num_elements)
{
    // strip off any filename prefix starting with a colon
    {
        const char *pos = strchr(file_name, ':');
        if (pos) file_name = pos + 1;
    }

    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("H5Fcreate failed: %lld\n", (long long)file_id);
        return false;
    }

    hsize_t dims[1];
    dims[0] = num_elements;
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    if (dataspace_id < 0) {
        printf("H5Screate_simple failed: %lld\n", (long long)dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    hid_t loc_id = file_id;
    std::vector<hid_t> group_ids;
    // leading slash in dataset path is optional - ignore if present
    if (*dataset_name == '/') dataset_name++;
    while (true) {
        const char *pos = strchr(dataset_name, '/');
        if (!pos) break;
        char *group_name = strndup(dataset_name, pos - dataset_name);
        hid_t id = H5Gcreate2(loc_id, group_name,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0) {
            printf("H5Gcreate2 failed: %lld\n", (long long)id);
            for (std::vector<hid_t>::const_iterator it = group_ids.begin(); it != group_ids.end(); ++it) {
                H5Gclose(*it);
            }
            H5Sclose(dataspace_id);
            H5Fclose(file_id);
            return false;
        }
        group_ids.push_back(id);
        loc_id = id;
        dataset_name = pos + 1;
    }

    hid_t dataset = H5Dcreate2(loc_id, dataset_name,
                    H5T_IEEE_F64LE, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        printf("H5Dcreate2 failed: %lld\n", (long long)dataset);
        for(std::vector<hid_t>::const_iterator it = group_ids.begin(); it != group_ids.end(); ++it) {
            H5Gclose(*it);
        }
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // close things up - attach will reopen later
    H5Dclose(dataset);
    for(std::vector<hid_t>::const_iterator it = group_ids.begin(); it != group_ids.end(); ++it) {
        H5Gclose(*it);
    }
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return true;
}


void initBaseGrid(Context& ctx, Runtime* rt, const int fs_size, LogicalRegion& region_of_fields, LogicalPartition& patches)
{
    Box2D grid_bounds = Box2D(Point2D(0,0), Point2D(NUM_PATCHES_X*PATCH_SIZE-1, NUM_PATCHES_Y*PATCH_SIZE-1));
    IndexSpace grid_isp = rt->create_index_space(ctx, grid_bounds);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator field_allocator = rt->create_field_allocator(ctx, fs);
    for (int i = 0; i < fs_size; i++) {
        const int fid = field_allocator.allocate_field(sizeof(Real), i);
        assert(fid == i);
    }
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


//template<typename Op>
//void elem_op_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
//    Op op_functor = task->args;
//}

void set_initial_condition_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {

    const PhysicalRegion& rgn_cvars = rgns[0];
    const PhysicalRegion& rgn_pvars = rgns[1];
    Box2D isp_domain = rt->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    // Set primitive variables
    {
        const FieldAccessor<WRITE_DISCARD, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
        const FieldAccessor<WRITE_DISCARD, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
        const FieldAccessor<WRITE_DISCARD, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
        const FieldAccessor<WRITE_DISCARD, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
        for (PointInBox2D ij(isp_domain); ij(); ij++) {
            rho[*ij] = 1.0;
            u  [*ij] = 0.0;
            v  [*ij] = 0.0;
            T  [*ij] = 1.0;
        }
    }

    // Convert primitive variables to conservative variables
    {
        const FieldAccessor<READ_ONLY, Real, 2> rho(rgn_pvars, static_cast<int>(PVARS_ID::DENSITY));
        const FieldAccessor<READ_ONLY, Real, 2> u  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_X)  );
        const FieldAccessor<READ_ONLY, Real, 2> v  (rgn_pvars, static_cast<int>(PVARS_ID::VEL_Y)  );
        const FieldAccessor<READ_ONLY, Real, 2> T  (rgn_pvars, static_cast<int>(PVARS_ID::TEMP)   );
        const FieldAccessor<WRITE_DISCARD, Real, 2> mass (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MASS_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> mmtx (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTX_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> mmty (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::MMTY_0));
        const FieldAccessor<WRITE_DISCARD, Real, 2> enrg (rgn_pvars, static_cast<int>(STAGE3_CVARS_ID::ENRG_0));
        for (PointInBox2D ij(isp_domain); ij(); ij++) {
            mass[*ij] = rho[*ij];
            mmtx[*ij] =   u[*ij];
            mmty[*ij] =   v[*ij];
            enrg[*ij] =   T[*ij];
        }
    }
}


void calc_rhs_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    constexpr int OVARS_RGN_ID = 0;
    constexpr int CVARS_RGN_ID = 1;
    constexpr int PVARS_RGN_ID = 2;
    constexpr int BVARS_RGN_ID = 3;

    // TODO: for now only set zeros
    Real* args = reinterpret_cast<Real*>(task->args);
    const Real arg0 = args[0];

}



void ssp_rk3_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    // TODO: Launch calc_rhs_task 3 times but using different stages of CVARS
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;

    LogicalRegion    rgn_coord;
    LogicalRegion    rgn_cvars;
    LogicalRegion    rgn_pvars;
    LogicalRegion    rgn_bvars;
    LogicalPartition patches_coord;
    LogicalPartition patches_cvars;
    LogicalPartition patches_pvars;
    LogicalPartition patches_bvars;

    printf("Call \"initBaseGrid\" from \"top_level_task\" ... ");
    initBaseGrid(ctx, rt, static_cast<int>(STAGE3_CVARS_ID::SIZE), rgn_cvars, patches_cvars);
    initBaseGrid(ctx, rt, static_cast<int>(       PVARS_ID::SIZE), rgn_pvars, patches_pvars);
    initBaseGrid(ctx, rt, static_cast<int>(       BVARS_ID::SIZE), rgn_bvars, patches_bvars);
    initBaseGrid(ctx, rt, static_cast<int>(       COORD_ID::SIZE), rgn_coord, patches_coord);
    printf("Done!\n");

    printf("\n\nSet initial conditions ... ");
    // TODO: index launch "set_initial_condition_task"
    printf("Done!\n");

    printf("\n\nStart time integration:\n");
    printf("Simulation done!!!\n");

    // Prepare for copy
    FieldSpace cp_fs = rt->create_field_space(ctx);
    {
        FieldAllocator allocator = rt->create_field_allocator(ctx, cp_fs);
        allocator.allocate_field(sizeof(Real), static_cast<int>(COPY_ID::FID_CP));
    }
    LogicalRegion cp_lr = rt->create_logical_region(ctx, rgn_coord.get_index_space(), cp_fs);

    char hdf5_file_name[256];
    char hdf5_dataset_name[256];
    strcpy(hdf5_file_name, "filename");
    strcpy(hdf5_dataset_name, "FID_CP");

    PhysicalRegion cp_pr;

    // create the HDF5 file first - attach wants it to already exist
    int num_elements = 1000;
    bool ok = generate_hdf_file(hdf5_file_name, hdf5_dataset_name, num_elements);
    assert(ok);
    std::map<FieldID,const char*> field_map;
    field_map[static_cast<Legion::FieldID>(PVARS_ID::VEL_X)] = hdf5_dataset_name;
    printf("Checkpointing data to HDF5 file '%s' (dataset='%s')\n",
            hdf5_file_name, hdf5_dataset_name);
    AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, rgn_cvars, rgn_cvars);
    al.attach_hdf5(hdf5_file_name, field_map, LEGION_FILE_READ_WRITE);
    cp_pr = rt->attach_external_resource(ctx, al);

    CopyLauncher copy_launcher;
    copy_launcher.add_copy_requirements(
        RegionRequirement(rgn_coord, READ_ONLY, EXCLUSIVE, rgn_coord),
        RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
    copy_launcher.add_src_field(0, static_cast<Legion::FieldID>(PVARS_ID::VEL_X));
    copy_launcher.add_dst_field(0, static_cast<Legion::FieldID>(COPY_ID::FID_CP));
    rt->issue_copy_operation(ctx, copy_launcher);

    Future f = rt->detach_external_resource(ctx, cp_pr);
    f.get_void_result(true /*silence warnings*/);
}




int main(int argc, char* argv[]) {

    printf("Hello World from the real main test!\n");

    Legion::Runtime::set_top_level_task_id(TASK_ID::TOP_LEVEL);

    {
        Legion::TaskVariantRegistrar registrar(TASK_ID::TOP_LEVEL, "top_level_task");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
    }

    return Legion::Runtime::start(argc, argv);

}



