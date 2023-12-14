import "regent"

require("fields")
local usr_config = require("input")
local grid = require("grid")

local problem_config = {}


-- INITIAL CONDITION --
task problem_config.setInitialCondition(
    grid_patch : region(ispace(int3d), grid_fsp),
    meta_patch : region(ispace(int1d), grid_meta_fsp),
    data_patch : region(ispace(int3d), PVARS)
)
where
    reads (grid_patch, meta_patch),
    writes (data_patch)
do
    -- todo
end

return problem_config