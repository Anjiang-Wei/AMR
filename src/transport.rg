import "regent"
local cmath      = terralib.includec("math.h")

config = require("input")
eos    = require("eos")

local trans = {}

trans.T_ref    = config.transport.T_ref;
trans.mu_ref   = config.transport.mu_ref;
trans.visc_exp = config.transport.visc_exp;
trans.Pr       = config.transport.Pr;

local cp = eos.Rg * eos.gamma / (eos.gamma - 1.0);

terra trans.calcDynVisc(T: double, p: double): double
    return trans.mu_ref * cmath.pow(trans.T_ref, trans.visc_exp)
end



terra trans.calcThermCond(T : double, p : double) : double
    return cp * trans.calcDynVisc(T, p) / trans.Pr;
end

return trans