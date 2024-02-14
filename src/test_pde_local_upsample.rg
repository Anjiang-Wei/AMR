import "regent"
solver = require("navier_stokes_local_upsample")
local target = os.getenv("OBJNAME")
local link_flags = terralib.newlist({"-lm"})
regentlib.saveobj(solver.main, target, "executable", nil, link_flags)
