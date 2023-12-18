import "regent"
local c      = regentlib.c

require("fields")
local usr_config = require("input")
local grid       = require("grid")
local eos        = require("eos")
local cmath      = terralib.includec("math.h")

local problem_config = {}


-- INITIAL CONDITION --
local alp  = 1.2
local eps  = 0.3
local U0   = 0.5
local T0   = 1.0 / (eos.Rg * eos.gamma)
local AT   = (eos.gamma - 1.0) * eps * eps / (4.0 * alp * eos.gamma);
local isen = 1.0 / (eos.gamma - 1.0);

task problem_config.setInitialCondition(
    grid_patch : region(ispace(int3d), grid_fsp),
    meta_patch : region(ispace(int1d), grid_meta_fsp),
    data_patch : region(ispace(int3d), PVARS)
)
where
    reads (grid_patch, meta_patch),
    writes (data_patch)
do
    for cij in grid_patch.ispace do
        var r : double = cmath.sqrt(grid_patch[cij].x * grid_patch[cij].x + grid_patch[cij].y * grid_patch[cij].y);
        var G : double = cmath.exp(alp * (1.0 - r * r));
        data_patch[cij].u   = U0 + r * eps * G * ( grid_patch[cij].y) / (r + double(r*r < 1e-30));
        data_patch[cij].v   =      r * eps * G * (-grid_patch[cij].x) / (r + double(r*r < 1e-30));
        data_patch[cij].T   = T0 - AT * G * G;
        data_patch[cij].rho = cmath.pow(1.0 - AT * G * G / T0, isen);
    end
end

return problem_config
