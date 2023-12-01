import "regent"

local numerics = {}

-- Collocated first derivative operator
--     x[i-2]   x[i-1]   x[i]     x[i+1]   x[i+2]
--       o--------o--------o--------o--------o
--      fm2      fm1       ^       fp1      fp2
terra numerics.der1Coll(fm2 : double, fm1 : double, f0 : double, fp1 : double, fp2 : double, inv_dx : double) : double
    return inv_dx * ((2.0/3.0) * (fp1 - fm1) - (1.0/12.0) * (fp2 - fm2));
end



-- Staggered first derivative operator
--     x[i-3/2]  x[i-1/2]  x[i+1/2]  x[i+3/2]
--   |----o----|----o----|----o----|----o----|
--       fm2       fm1   ^   fp1       fp2 
terra numerics.der1Stag(fm2 : double, fm1 : double, fp1 : double, fp2 : double, inv_dx : double) : double
    return inv_dx * ((9.0/8.0) * (fp1 - fm1) - (1.0/24.0) * (fp2 - fm2));
end



-- Mid-point interpolation operator
--     x[i-3/2]  x[i-1/2]  x[i+1/2]  x[i+3/2]
--   |----o----|----o----|----o----|----o----|
--       fm2       fm1   ^   fp1       fp2 
terra numerics.midInterp(fm2 : double, fm1 : double, fp1 : double, fp2 : double) : double
    return (9.0/16.0) * (fp1 + fm1) - (1.0/16.0) * (fp2 + fm2);
end

return numerics

