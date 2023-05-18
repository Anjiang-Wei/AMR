
#include "hdf5_hook.h"


void writeFieldsToH5(const std::string filename, const std::vector<std::string> fnames, const std::vector<uint> fids, RegionOfFields rgn, BaseGridConfig g_config) {
    hid_t file_id = H5Fcreate((filename + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[2] = {g_config.PATCH_SIZE * g_config.NUM_PATCHES_X, g_config.PATCH_SIZE * g_config.NUM_PATCHES_Y};
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

    const int num_fields = fnames.size();
    for (int i = 0; i < num_fields; i++) {
        hid_t dset_id = H5Dcreate2(file_id, ("/"+fnames[i]).c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dset_id);
    }
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

}

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
