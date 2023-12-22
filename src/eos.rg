import "regent"
config = require("input")


local eos_model = {}

eos_model.gamma = config.eos.gamma
eos_model.Rg    = config.eos.Rg

local cv = eos_model.Rg / (eos_model.gamma - 1.0);

terra eos_model.calcInternalEnergy(T : double, p : double) : double
    return cv * T;
end

return eos_model
