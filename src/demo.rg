import "regent"
local c = regentlib.c

function whatever()
    local task empty()
    end
    return empty
end

task main()
    [whatever()]() -- 1st time call
    [whatever()]() -- 2nd time call
end

local target = os.getenv("OBJNAME")
regentlib.saveobj(main, target, "executable")
