import "regent"
solver = require("riemann_local_upsample")
local target = os.getenv("OBJNAME")
local link_flags = terralib.newlist({"-lm"})
regentlib.saveobj(solver.main, target, "executable", nil, link_flags)
