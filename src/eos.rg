import "regent"
config = require("input")


local eos_model = {}

eos_model.gamma = config.eos.gamma
eos_model.Rg    = config.eos.Rg

terra eos_model.calcInternalEnergy(T : double, p : double) : double
    return double(eos_model.Rg) * T / (double(eos_model.gamma) - 1.0);
end

return eos_model
