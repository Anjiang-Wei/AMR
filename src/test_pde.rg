import "regent"
solver = require("navier_stokes")
local target = os.getenv("OBJNAME")
local link_flags = terralib.newlist({"-lm"})
regentlib.saveobj(solver.main, target, "executable", nil, link_flags)
