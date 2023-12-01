import "regent"
solver = require("navier_stokes")
local target = os.getenv("OBJNAME")
regentlib.saveobj(solver.main, target, "executable")

