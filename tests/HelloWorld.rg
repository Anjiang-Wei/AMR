import "regent"

local format = require("std/format")

task hello_world()
    format.println("Hello World!")
end

task main()
    hello_world()
end

local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")

