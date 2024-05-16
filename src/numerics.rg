import "regent"
local cmath = terralib.includec("math.h")

local numerics = {}

numerics.stencil_width     = 5;
numerics.stencil_width_ext = 9;


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


-- Upsampling interpolation operator to get the solution on the left
--     x[i-2]  x[i-1]   x[i]   x[i+1]  x[i+2]
--                   |-o-|---|
--   |---o---|---o---|---o---|---o---|---o---|
--      fm2     fm1      ^      fp1     fp2
terra numerics.upSample(fm2 : double, fm1 : double, f00 : double, fp1 : double, fp2 : double) : double
    return -(45./2048) * fm2 + (105./512.) * fm1 + (945./1024.) * f00 - (63./512) * fp1 + (35./2048.) * fp2
end



 -- 5th-order WENO5-Z interpolation scheme.
 -- For more detail see Jiang & Shu, JCP (1996).
 -- Note: The coefficients used are based on interpolation not reconstruction.
 
 --          j-2     j-1      j      j+1     j+2     j+3
 --           |-------|-------|---x---|-------|-------|
 
 --    Fwd interpolation:
 --          fm2     fm1     f00     fp1     fp2
 --        S0 o-------o-------o
 --                S1 o-------o-------o
 --                        S2 o-------o-------o
 
 --    Bwd interpolation:
 --                 fp2      fp1     f00     fm1     fm2
 --                                   o-------o-------o S0
 --                           o-------o-------o S1
 --                   o-------o-------o S2
 terra numerics.WENO5Z(fm2 : double, fm1 : double, f00 : double, fp1 : double, fp2 : double) : double
    var q0  : double = 0.375 * fm2 - 1.250 * fm1 + 1.875 * f00;
    var q1  : double =-0.125 * fm1 + 0.750 * f00 + 0.375 * fp1;
    var q2  : double = 0.375 * f00 + 0.750 * fp1 - 0.125 * fp2;
    var d20 : double = fm2 - 2.0*fm1 + f00;
    var d21 : double = fm1 - 2.0*f00 + fp1;
    var d22 : double = f00 - 2.0*fp1 + fp2;
    var d10 : double = fm2 - 4.0*fm1 + 3.0*f00;
    var d11 : double = fm1 - fp1;
    var d12 : double = fp2 - 4.0*fp1 + 3.0*f00;
    var IS0 : double = (13.0/12.0) * d20 * d20 + 0.25 * d10 * d10 + 1e-6;
    var IS1 : double = (13.0/12.0) * d21 * d21 + 0.25 * d11 * d11 + 1e-6;
    var IS2 : double = (13.0/12.0) * d22 * d22 + 0.25 * d12 * d12 + 1e-6;
    var tau2: double = (IS0 - IS2) * (IS0 - IS2);
    var a0  : double = 0.0625 * (1.0 + tau2 / (IS0 * IS0));
    var a1  : double = 0.6250 * (1.0 + tau2 / (IS1 * IS1));
    var a2  : double = 0.3125 * (1.0 + tau2 / (IS2 * IS2));
    return  (a0*q0 + a1*q1 + a2*q2) / (a0+a1+a2+1e-32);
end


struct numerics.CharVars {
    var0 : double,
    var1 : double,
    var2 : double,
    var3 : double
}


terra numerics.charDecompGetCharVarsX(prim_vars : numerics.CharVars, u : double, v : double, gamma : double, c : double) : numerics.CharVars
    var char_vars : numerics.CharVars;
    var gm1byc2 : double = 0.5 * (gamma - 1.0) / (c * c);
    var ek : double = 0.5 * (u * u + v * v);

    char_vars.var0 = prim_vars.var0 * (gm1byc2 * ek + 0.5 * u / c)
                   + prim_vars.var1 * (- gm1byc2 * u - 0.5 / c)
                   + prim_vars.var2 * (- gm1byc2 * v)
                   + prim_vars.var3 * gm1byc2;

    char_vars.var1 = prim_vars.var0 * (-v)
                   + prim_vars.var2;

    char_vars.var2 = prim_vars.var0 * (1.0 - 2.0 * gm1byc2 * ek)
                   + prim_vars.var1 * (2.0 * gm1byc2 * u)
                   + prim_vars.var2 * (2.0 * gm1byc2 * v)
                   + prim_vars.var3 * (-2.0 * gm1byc2);

    char_vars.var3 = prim_vars.var0 * (gm1byc2 * ek - 0.5 * u / c)
                   + prim_vars.var1 * (- gm1byc2 * u + 0.5 / c)
                   + prim_vars.var2 * (- gm1byc2 * v)
                   + prim_vars.var3 * gm1byc2;

    return char_vars;
end


terra numerics.charDecompGetCharVarsY(prim_vars : numerics.CharVars, u : double, v : double, gamma : double, c : double) : numerics.CharVars
    var char_vars : numerics.CharVars;
    var gm1byc2 : double = 0.5 * (gamma - 1.0) / (c * c);
    var ek : double = 0.5 * (u * u + v * v);

    char_vars.var0 = prim_vars.var0 * (gm1byc2 * ek + 0.5 * v / c)
                   + prim_vars.var1 * (- gm1byc2 * u)
                   + prim_vars.var2 * (- gm1byc2 * v - 0.5 / c)
                   + prim_vars.var3 * gm1byc2;

    char_vars.var1 = prim_vars.var0 * (-u)
                   + prim_vars.var1;

    char_vars.var2 = prim_vars.var0 * (1.0 - 2.0 * gm1byc2 * ek)
                   + prim_vars.var1 * (2.0 * gm1byc2 * u)
                   + prim_vars.var2 * (2.0 * gm1byc2 * v)
                   + prim_vars.var3 * (-2.0 * gm1byc2);

    char_vars.var3 = prim_vars.var0 * (gm1byc2 * ek - 0.5 * u / c)
                   + prim_vars.var1 * (- gm1byc2 * u)
                   + prim_vars.var2 * (- gm1byc2 * v + 0.5 / c)
                   + prim_vars.var3 * gm1byc2;

    return char_vars;
end


terra numerics.charDecompGetConsVarsX(char_vars : numerics.CharVars, u : double, v : double, h : double, gamma : double) : numerics.CharVars
    var cons_vars : numerics.CharVars;
    var ek : double = 0.5 * (u * u + v * v);
    var c : double = cmath.sqrt((gamma - 1.0) * (h - ek));

    cons_vars.var0 = char_vars.var0 + char_vars.var2 + char_vars.var3;
    
    cons_vars.var1 = char_vars.var0 * (u - c)
                   + char_vars.var2 * u
                   + char_vars.var3 * (u + c);

    cons_vars.var2 = char_vars.var0 * v
                   + char_vars.var1
                   + char_vars.var2 * v
                   + char_vars.var3 * v;

    cons_vars.var3 = char_vars.var0 * (h - c * u)
                   + char_vars.var1 * v
                   + char_vars.var2 * ek
                   + char_vars.var3 * (h + c * u);

    return cons_vars;
end


terra numerics.charDecompGetConsVarsY(char_vars : numerics.CharVars, u : double, v : double, h : double, gamma : double) : numerics.CharVars
    var cons_vars : numerics.CharVars;
    var ek : double = 0.5 * (u * u + v * v);
    var c : double = cmath.sqrt((gamma - 1.0) * (h - ek));

    cons_vars.var0 = char_vars.var0 + char_vars.var2 + char_vars.var3;
    
    cons_vars.var1 = char_vars.var0 * u
                   + char_vars.var1
                   + char_vars.var2 * u
                   + char_vars.var3 * u;

    cons_vars.var2 = char_vars.var0 * (v - c)
                   + char_vars.var2 * v
                   + char_vars.var3 * (v + c);

    cons_vars.var3 = char_vars.var0 * (h - c * v)
                   + char_vars.var1 * u
                   + char_vars.var2 * ek
                   + char_vars.var3 * (h + c * v);

    return cons_vars;
end


terra numerics.RiemannLF(S : double, UL : numerics.CharVars, UR : numerics.CharVars, FL : numerics.CharVars, FR : numerics.CharVars) : numerics.CharVars
    var FRL : numerics.CharVars;
    FRL.var0 = 0.5 * (FR.var0 + FL.var0) - 0.5 * S * (UR.var0 - UL.var0);
    FRL.var1 = 0.5 * (FR.var1 + FL.var1) - 0.5 * S * (UR.var1 - UL.var1);
    FRL.var2 = 0.5 * (FR.var2 + FL.var2) - 0.5 * S * (UR.var2 - UL.var2);
    FRL.var3 = 0.5 * (FR.var3 + FL.var3) - 0.5 * S * (UR.var3 - UL.var3);
    return FRL
end

return numerics

