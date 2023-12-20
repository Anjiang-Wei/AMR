import "regent"


local eos_model = {}

eos_model.gamma = 1.4
eos_model.Rg    = 1.0

terra eos_model.foo(a : double, b : double) : double
    return a + b;
end

return eos_model
