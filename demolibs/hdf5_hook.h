
#ifndef _HDF5_HOOK_H
#define _HDF5_HOOK_H

#include <vector>
#include <string>
#include <hdf5.h>
#include "util.h"
#include "legion_hook.h"

enum class HDF5_COPY_ID {
    FID_CP,
};

void writeFieldsToH5(const std::string, const std::vector<std::string>, const std::vector<uint>, RegionOfFields, BaseGridConfig);

bool writeFieldsToH5(const std::string);

bool generate_hdf_file(const char*, const char*, int);

#endif
