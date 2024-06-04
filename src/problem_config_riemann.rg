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
local T0   = 1.0 / (eos.Rg * eos.gamma)
local U0   = 0.5 * cmath.sqrt(eos.gamma * eos.Rg * T0)
local AT   = (eos.gamma - 1.0) * eps * eps / (4.0 * alp * eos.gamma);
local isen = 1.0 / (eos.gamma - 1.0);

__demand(__leaf, __inline)
task problem_config.setInitialCondition(
    grid_patch : region(ispace(int3d), grid_fsp),
    meta_patch : region(ispace(int1d), grid_meta_fsp),
    data_patch : region(ispace(int3d), CVARS)
)
where
    reads (grid_patch, meta_patch),
    writes (data_patch)
do
    for cij in grid_patch.ispace do
        var xyp : double = grid_patch[cij].x + grid_patch[cij].y;
        var xym : double = grid_patch[cij].x - grid_patch[cij].y;

        var rho : double;
        var p   : double;

        if (xyp * xyp < 0.0225 and xym * xym < 0.0225) then
            rho = 0.125;
            p   = 0.140;
        else
            rho = 1.000;
            p   = 1.000;
        end

        var T   : double = p / (eos.Rg * rho); 
        var u   : double = 0.0;
        var v   : double = 0.0;

        data_patch[cij].mass = rho;
        data_patch[cij].mmtx = rho * u;
        data_patch[cij].mmty = rho * v;
        data_patch[cij].enrg = rho * (double(eos.Rg) / double(eos.gamma - 1.0) * T + 0.5 * (u * u + v * v));
    end
end


return problem_config
