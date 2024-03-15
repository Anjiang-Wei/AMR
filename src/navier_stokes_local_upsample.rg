import "regent"
local usr_config     = require("input")
local problem_config = require("problem_config")
local c              = regentlib.c
local stdlib         = terralib.includec("stdlib.h")
local string         = terralib.includec("string.h")
local cmath          = terralib.includec("math.h")
local format         = require("std/format")
local grid           = require("grid")
local numerics       = require(usr_config.numerics_modules)
local eos            = require("eos")
local trans          = require("transport")
require("fields")

local domain_length_x         = usr_config.domain_length_x
local domain_length_y         = usr_config.domain_length_y
local domain_shift_x          = usr_config.domain_shift_x
local domain_shift_y          = usr_config.domain_shift_y 
local patch_size              = usr_config.patch_size
local num_base_patches_i      = usr_config.num_base_patches_i
local num_base_patches_j      = usr_config.num_base_patches_j

local num_grid_points_base_i  = num_base_patches_i * patch_size
local num_grid_points_base_j  = num_base_patches_j * patch_size

local solver = {}

local
terra pow2(e : int) : int
    return 1 << e
end


-- Calculate coordinate of grid points with given patch coordinate and level
task solver.setGridPointCoordinates(
    grid_patch : region(ispace(int3d), grid_fsp),
    meta_patch : region(ispace(int1d), grid_meta_fsp)
)
where
    reads (meta_patch.{i_coord, j_coord, level}),
    writes (grid_patch)
do
    var level_fact  = pow2(meta_patch[meta_patch.bounds.lo].level)
    var dx : double = double(domain_length_x) / (num_grid_points_base_i * level_fact)
    var dy : double = double(domain_length_y) / (num_grid_points_base_j * level_fact)
    var color       = meta_patch.bounds.lo;
    var x_start: double = meta_patch[color].i_coord * patch_size * dx + domain_shift_x;
    var y_start: double = meta_patch[color].j_coord * patch_size * dy + domain_shift_y;
    for cij in grid_patch.ispace do
        grid_patch[cij].x = x_start + dx * cij.y
        grid_patch[cij].y = y_start + dy * cij.z
    end
end


-- Calculate velocity gradient tensor at collocated points
__demand(__leaf, __inline)
task solver.calcGradVelColl(
    grad_vel_coll_patch : region(ispace(int3d), GRAD_VEL),      -- patch that only contains the interior region
    c_vars_now_patch    : region(ispace(int3d), CVARS),         -- patch including the halo layer on each side
    meta_patch          : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(grad_vel_coll_patch),
    reads(c_vars_now_patch.{mass, mmtx, mmty, enrg}, meta_patch.{level})
do
    var level_fact  = pow2(meta_patch[meta_patch.bounds.lo].level)    
    var inv_dx : double = 1.0 / (double(domain_length_x) / (num_grid_points_base_i * level_fact));
    var inv_dy : double = 1.0 / (double(domain_length_y) / (num_grid_points_base_j * level_fact));
    var stencil_buf_u : double[numerics.stencil_width];
    var stencil_buf_v : double[numerics.stencil_width];
    var stencil_idx_shift : int = (numerics.stencil_width - 1) / 2;
    for cij in grad_vel_coll_patch.ispace do
        for m = 0, numerics.stencil_width do
            var cij_shifted : int3d = int3d({cij.x, cij.y + m - stencil_idx_shift, cij.z});
            stencil_buf_u[m] = c_vars_now_patch[cij_shifted].mmtx / c_vars_now_patch[cij_shifted].mass;
            stencil_buf_v[m] = c_vars_now_patch[cij_shifted].mmty / c_vars_now_patch[cij_shifted].mass;
        end
        grad_vel_coll_patch[cij].dudx = numerics.der1Coll(stencil_buf_u[0], stencil_buf_u[1], stencil_buf_u[2], stencil_buf_u[3], stencil_buf_u[4], inv_dx);
        grad_vel_coll_patch[cij].dvdx = numerics.der1Coll(stencil_buf_v[0], stencil_buf_v[1], stencil_buf_v[2], stencil_buf_v[3], stencil_buf_v[4], inv_dx);

        for m = 0, numerics.stencil_width do
            var cij_shifted : int3d = int3d({cij.x, cij.y, cij.z + m - stencil_idx_shift});
            stencil_buf_u[m] = c_vars_now_patch[cij_shifted].mmtx / c_vars_now_patch[cij_shifted].mass;
            stencil_buf_v[m] = c_vars_now_patch[cij_shifted].mmty / c_vars_now_patch[cij_shifted].mass;
        end
        grad_vel_coll_patch[cij].dudy = numerics.der1Coll(stencil_buf_u[0], stencil_buf_u[1], stencil_buf_u[2], stencil_buf_u[3], stencil_buf_u[4], inv_dy);
        grad_vel_coll_patch[cij].dvdy = numerics.der1Coll(stencil_buf_v[0], stencil_buf_v[1], stencil_buf_v[2], stencil_buf_v[3], stencil_buf_v[4], inv_dy);
    end
end


local struct PrimitiveVars {
    u : double,
    v : double,
    T : double,
    p : double
}

local struct ConservativeVars {
    mass : double,
    mmtx : double,
    mmty : double,
    enrg : double
}

local terra conservativeToPrimitive (rho : double, rho_u : double, rho_v : double, rho_e : double) : PrimitiveVars
    var pvars : PrimitiveVars;
    pvars.u = rho_u / rho;
    pvars.v = rho_v / rho;
    pvars.T = ((rho_e / rho) - 0.5 * (pvars.u * pvars.u + pvars.v * pvars.v)) / double(eos.Rg) * (eos.gamma - 1.0);
    pvars.p = rho * eos.Rg * pvars.T;
    return pvars;
end 

local terra primitiveToConservative (u : double, v : double, T : double, p : double) : ConservativeVars
    var cvars : ConservativeVars;
    var rho : double = p / (eos.Rg * T);
    cvars.mass = rho;
    cvars.mmtx = rho * u;
    cvars.mmty = rho * v;
    cvars.enrg = p / (eos.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    return cvars;
end

-- Calculate the right-hand side of the NS equations
__demand(__leaf, __inline)
task solver.calcRHSLeaf(
    c_vars_ddt_patch    : region(ispace(int3d), CVARS   ),     -- patch that only contains the interior region
    c_vars_now_patch    : region(ispace(int3d), CVARS   ),     -- patch including the halo layer on each side
    grad_vel_coll_patch : region(ispace(int3d), GRAD_VEL),     -- patch including the halo layer on each side
    grid_patch          : region(ispace(int3d), grid_fsp),     -- patch including the halo layer on each side
    meta_patch          : region(ispace(int1d), grid_meta_fsp)
)
where
    writes(c_vars_ddt_patch),
    reads (c_vars_now_patch, grad_vel_coll_patch, grid_patch, meta_patch.level)
do

    var    u_coll : double [numerics.stencil_width_ext];
    var    v_coll : double [numerics.stencil_width_ext];
    var    T_coll : double [numerics.stencil_width_ext];
    var    p_coll : double [numerics.stencil_width_ext];
    var dudx_coll : double [numerics.stencil_width_ext];
    var dudy_coll : double [numerics.stencil_width_ext];
    var dvdx_coll : double [numerics.stencil_width_ext];
    var dvdy_coll : double [numerics.stencil_width_ext];

    var flux_mass : double [numerics.stencil_width - 1];
    var flux_mmtx : double [numerics.stencil_width - 1];
    var flux_mmty : double [numerics.stencil_width - 1];
    var flux_enrg : double [numerics.stencil_width - 1];

    var stencil_ctr        : int = (numerics.stencil_width     - 1) / 2;
    var stencil_ext_ctr    : int = (numerics.stencil_width_ext - 1) / 2;
    var stencil_coll_shift : int = stencil_ext_ctr - stencil_ctr;

    var level_fact  = pow2(meta_patch[meta_patch.bounds.lo].level)    
    var inv_dx : double = 1.0 / (double(domain_length_x) / (num_grid_points_base_i * level_fact));
    var inv_dy : double = 1.0 / (double(domain_length_y) / (num_grid_points_base_j * level_fact));

    for cij in c_vars_ddt_patch.ispace do
        -----------------------
        -- ASSEMBLE X-FLUXES --
        -----------------------
        for i = 0, numerics.stencil_width_ext do
            var cij_shifted : int3d = int3d({cij.x, cij.y - stencil_ext_ctr + i, cij.z});
            var pvars : PrimitiveVars = conservativeToPrimitive(c_vars_now_patch[cij_shifted].mass, c_vars_now_patch[cij_shifted].mmtx, c_vars_now_patch[cij_shifted].mmty, c_vars_now_patch[cij_shifted].enrg);
            u_coll[i] = pvars.u;
            v_coll[i] = pvars.v;
            T_coll[i] = pvars.T;
            p_coll[i] = pvars.p;
            dudy_coll[i] = grad_vel_coll_patch[cij_shifted].dudy
            dvdy_coll[i] = grad_vel_coll_patch[cij_shifted].dvdy
        end

        for i = 0, (numerics.stencil_width - 1) do
            var i_coll : int = i + stencil_coll_shift;
            var u : double = numerics.midInterp(u_coll[i_coll - 1], u_coll[i_coll], u_coll[i_coll + 1], u_coll[i_coll + 2]);
            var v : double = numerics.midInterp(v_coll[i_coll - 1], v_coll[i_coll], v_coll[i_coll + 1], v_coll[i_coll + 2]);
            var T : double = numerics.midInterp(T_coll[i_coll - 1], T_coll[i_coll], T_coll[i_coll + 1], T_coll[i_coll + 2]);
            var p : double = numerics.midInterp(p_coll[i_coll - 1], p_coll[i_coll], p_coll[i_coll + 1], p_coll[i_coll + 2]);

            var dudx : double = numerics.der1Stag (u_coll[i_coll - 1], u_coll[i_coll], u_coll[i_coll + 1], u_coll[i_coll + 2], inv_dx);
            var dvdx : double = numerics.der1Stag (v_coll[i_coll - 1], v_coll[i_coll], v_coll[i_coll + 1], v_coll[i_coll + 2], inv_dx);
            var dTdx : double = numerics.der1Stag (T_coll[i_coll - 1], T_coll[i_coll], T_coll[i_coll + 1], T_coll[i_coll + 2], inv_dx);
            var dudy : double = numerics.midInterp(dudy_coll[i_coll - 1], dudy_coll[i_coll], dudy_coll[i_coll + 1], dudy_coll[i_coll + 2]);
            var dvdy : double = numerics.midInterp(dvdy_coll[i_coll - 1], dvdy_coll[i_coll], dvdy_coll[i_coll + 1], dvdy_coll[i_coll + 2]);

            var rho : double = p / (eos.Rg * T);
            var mu  : double = trans.calcDynVisc(T, p);
            var kap : double = trans.calcThermCond(T, p);
            var H   : double = rho * (eos.calcInternalEnergy(T, p) + 0.5 * (u * u + v * v)) + p;
            var st11: double = 2.0 * mu *  dudx - (2.0/3.0) * mu * (dudx + dvdy);
            var st21: double = 2.0 * mu * (dvdx + dudy);

            flux_mass[i] = -rho * u;
            flux_mmtx[i] = -rho * u * u - p + st11;
            flux_mmty[i] = -rho * v * u     + st21;
            flux_enrg[i] = -H   *     u     + u * st11 + v * st21 + kap * dTdx;
        end
        var flux_div_x_mass : double = numerics.der1Stag(flux_mass[0], flux_mass[1], flux_mass[2], flux_mass[3], inv_dx)
        var flux_div_x_mmtx : double = numerics.der1Stag(flux_mmtx[0], flux_mmtx[1], flux_mmtx[2], flux_mmtx[3], inv_dx)
        var flux_div_x_mmty : double = numerics.der1Stag(flux_mmty[0], flux_mmty[1], flux_mmty[2], flux_mmty[3], inv_dx)
        var flux_div_x_enrg : double = numerics.der1Stag(flux_enrg[0], flux_enrg[1], flux_enrg[2], flux_enrg[3], inv_dx)

        -----------------------
        -- ASSEMBLE Y-FLUXES --
        -----------------------
        for j = 0, numerics.stencil_width_ext do
            var cij_shifted : int3d = int3d({cij.x, cij.y, cij.z - stencil_ext_ctr + j});
            var pvars : PrimitiveVars = conservativeToPrimitive(c_vars_now_patch[cij_shifted].mass, c_vars_now_patch[cij_shifted].mmtx, c_vars_now_patch[cij_shifted].mmty, c_vars_now_patch[cij_shifted].enrg);
            u_coll[j] = pvars.u;
            v_coll[j] = pvars.v;
            T_coll[j] = pvars.T;
            p_coll[j] = pvars.p;
            dudx_coll[j] = grad_vel_coll_patch[cij_shifted].dudx
            dvdx_coll[j] = grad_vel_coll_patch[cij_shifted].dvdx
        end

        for j = 0, (numerics.stencil_width - 1) do
            var j_coll : int = j + stencil_coll_shift;
            var u : double = numerics.midInterp(u_coll[j_coll - 1], u_coll[j_coll], u_coll[j_coll + 1], u_coll[j_coll + 2]);
            var v : double = numerics.midInterp(v_coll[j_coll - 1], v_coll[j_coll], v_coll[j_coll + 1], v_coll[j_coll + 2]);
            var T : double = numerics.midInterp(T_coll[j_coll - 1], T_coll[j_coll], T_coll[j_coll + 1], T_coll[j_coll + 2]);
            var p : double = numerics.midInterp(p_coll[j_coll - 1], p_coll[j_coll], p_coll[j_coll + 1], p_coll[j_coll + 2]);

            var dudy : double = numerics.der1Stag (u_coll[j_coll - 1], u_coll[j_coll], u_coll[j_coll + 1], u_coll[j_coll + 2], inv_dy);
            var dvdy : double = numerics.der1Stag (v_coll[j_coll - 1], v_coll[j_coll], v_coll[j_coll + 1], v_coll[j_coll + 2], inv_dy);
            var dTdy : double = numerics.der1Stag (T_coll[j_coll - 1], T_coll[j_coll], T_coll[j_coll + 1], T_coll[j_coll + 2], inv_dy);
            var dudx : double = numerics.midInterp(dudx_coll[j_coll - 1], dudx_coll[j_coll], dudx_coll[j_coll + 1], dudx_coll[j_coll + 2]);
            var dvdx : double = numerics.midInterp(dvdx_coll[j_coll - 1], dvdx_coll[j_coll], dvdx_coll[j_coll + 1], dvdx_coll[j_coll + 2]);

            var rho : double = p / (eos.Rg * T);
            var mu  : double = trans.calcDynVisc(T, p);
            var kap : double = trans.calcThermCond(T, p);
            var H   : double = rho * (eos.calcInternalEnergy(T, p) + 0.5 * (u * u + v * v)) + p;
            var st12: double = 2.0 * mu * (dvdx + dudy);
            var st22: double = 2.0 * mu *  dvdy - (2.0/3.0) * mu * (dudx + dvdy);

            flux_mass[j] = -rho * v;
            flux_mmtx[j] = -rho * u * v     + st12;
            flux_mmty[j] = -rho * v * v - p + st22;
            flux_enrg[j] = -H   *     v     + u * st12 + v * st22 + kap * dTdy;
        end
        var flux_div_y_mass : double = numerics.der1Stag(flux_mass[0], flux_mass[1], flux_mass[2], flux_mass[3], inv_dy)
        var flux_div_y_mmtx : double = numerics.der1Stag(flux_mmtx[0], flux_mmtx[1], flux_mmtx[2], flux_mmtx[3], inv_dy)
        var flux_div_y_mmty : double = numerics.der1Stag(flux_mmty[0], flux_mmty[1], flux_mmty[2], flux_mmty[3], inv_dy)
        var flux_div_y_enrg : double = numerics.der1Stag(flux_enrg[0], flux_enrg[1], flux_enrg[2], flux_enrg[3], inv_dy)


        c_vars_ddt_patch[cij].mass = flux_div_x_mass + flux_div_y_mass
        c_vars_ddt_patch[cij].mmtx = flux_div_x_mmtx + flux_div_y_mmtx
        c_vars_ddt_patch[cij].mmty = flux_div_x_mmty + flux_div_y_mmty
        c_vars_ddt_patch[cij].enrg = flux_div_x_enrg + flux_div_y_enrg

    end -- for cij in c_vars_ddt_patch.ispace 

end


-- Compute algebraic operations in SSP-RK3 scheme
function SSPRK3Stage(stage)
    if      stage == 0 then
        -- u1 = u0 + u1 * dt
        local task ssprk3Stage(
            dt : double,
            u0 : region(ispace(int3d), CVARS),
            u1 : region(ispace(int3d), CVARS)
        )
        where
            reads (u0, u1),
            writes(u1)
        do
            var offset_0 = u0.bounds.lo - u1.bounds.lo;
            __demand(__vectorize)
            for cij in u1.ispace do
                u1[cij].mass *= dt
                u1[cij].mmtx *= dt
                u1[cij].mmty *= dt
                u1[cij].enrg *= dt
                u1[cij].mass += u0[cij + offset_0].mass
                u1[cij].mmtx += u0[cij + offset_0].mmtx
                u1[cij].mmty += u0[cij + offset_0].mmty
                u1[cij].enrg += u0[cij + offset_0].enrg
                --u1[cij].mass = u0[cij + offset_0].mass + u1[cij].mass * dt
                --u1[cij].mmtx = u0[cij + offset_0].mmtx + u1[cij].mmtx * dt
                --u1[cij].mmty = u0[cij + offset_0].mmty + u1[cij].mmty * dt
                --u1[cij].enrg = u0[cij + offset_0].enrg + u1[cij].enrg * dt
            end
        end
        return ssprk3Stage
    elseif stage == 1 then
        -- u2 = 0.75 * u0 + 0.25 * u1 + 0.25 * dt * u2
        local task ssprk3Stage(
            dt : double,
            u0 : region(ispace(int3d), CVARS),
            u1 : region(ispace(int3d), CVARS),
            u2 : region(ispace(int3d), CVARS)
        )
        where
            reads (u0, u1, u2),
            writes(u2)
        do
            var offset_0 = u0.bounds.lo - u2.bounds.lo
            var offset_1 = u1.bounds.lo - u2.bounds.lo
            __demand(__vectorize)
            for cij in u2.ispace do
                u2[cij].mass *= 0.25 * dt
                u2[cij].mmtx *= 0.25 * dt
                u2[cij].mmty *= 0.25 * dt
                u2[cij].enrg *= 0.25 * dt
                u2[cij].mass += 0.75 * u0[cij + offset_0].mass + 0.25 * u1[cij + offset_1].mass
                u2[cij].mmtx += 0.75 * u0[cij + offset_0].mmtx + 0.25 * u1[cij + offset_1].mmtx
                u2[cij].mmty += 0.75 * u0[cij + offset_0].mmty + 0.25 * u1[cij + offset_1].mmty
                u2[cij].enrg += 0.75 * u0[cij + offset_0].enrg + 0.25 * u1[cij + offset_1].enrg
                -- u2[cij].mass = 0.75 * u0[cij + offset_0].mass + 0.25 * (u1[cij + offset_1].mass + u2[cij].mass * dt)
                -- u2[cij].mmtx = 0.75 * u0[cij + offset_0].mmtx + 0.25 * (u1[cij + offset_1].mmtx + u2[cij].mmtx * dt)
                -- u2[cij].mmty = 0.75 * u0[cij + offset_0].mmty + 0.25 * (u1[cij + offset_1].mmty + u2[cij].mmty * dt)
                -- u2[cij].enrg = 0.75 * u0[cij + offset_0].enrg + 0.25 * (u1[cij + offset_1].enrg + u2[cij].enrg * dt)
            end
        end
        return ssprk3Stage
    elseif stage == 2 then
        -- u0 = (1/3) * u0 + (2/3) * u2 + (2/3) * u1 * dt 
        local task ssprk3Stage(
            dt : double,
            u0 : region(ispace(int3d), CVARS),
            u1 : region(ispace(int3d), CVARS),
            u2 : region(ispace(int3d), CVARS)
        )
        where
            reads(u0, u1, u2),
            writes(u0)
        do
            var offset_1 = u1.bounds.lo - u0.bounds.lo
            var offset_2 = u2.bounds.lo - u0.bounds.lo
            __demand(__vectorize)
            for cij in u0.ispace do
                u0[cij].mass *= 1.0 / 3.0
                u0[cij].mmtx *= 1.0 / 3.0
                u0[cij].mmty *= 1.0 / 3.0
                u0[cij].enrg *= 1.0 / 3.0
                u0[cij].mass += (2.0 / 3.0) * u2[cij + offset_2].mass + (2.0 / 3.0) * u1[cij + offset_1].mass * dt
                u0[cij].mmtx += (2.0 / 3.0) * u2[cij + offset_2].mmtx + (2.0 / 3.0) * u1[cij + offset_1].mmtx * dt
                u0[cij].mmty += (2.0 / 3.0) * u2[cij + offset_2].mmty + (2.0 / 3.0) * u1[cij + offset_1].mmty * dt
                u0[cij].enrg += (2.0 / 3.0) * u2[cij + offset_2].enrg + (2.0 / 3.0) * u1[cij + offset_1].enrg * dt
                -- u0[cij].mass = (1.0/3.0) * u0[cij].mass + (2.0/3.0) * (u2[cij + offset_2].mass + u1[cij + offset_1].mass * dt)
                -- u0[cij].mmtx = (1.0/3.0) * u0[cij].mmtx + (2.0/3.0) * (u2[cij + offset_2].mmtx + u1[cij + offset_1].mmtx * dt)
                -- u0[cij].mmty = (1.0/3.0) * u0[cij].mmty + (2.0/3.0) * (u2[cij + offset_2].mmty + u1[cij + offset_1].mmty * dt)
                -- u0[cij].enrg = (1.0/3.0) * u0[cij].enrg + (2.0/3.0) * (u2[cij + offset_2].enrg + u1[cij + offset_1].enrg * dt)
            end
        end
        return ssprk3Stage
    end
end



-- Inner task RHS
task solver.calcRHSLaunch(
     level                           : int,
     rgn_cvars_ddt                   : region(ispace(int3d), CVARS   ),
     rgn_cvars_now                   : region(ispace(int3d), CVARS   ),
     rgn_grad_vel_coll               : region(ispace(int3d), GRAD_VEL),
     rgn_grid                        : region(ispace(int3d), grid_fsp),    
     rgn_meta                        : region(ispace(int1d), grid_meta_fsp),
     --
     part_cvars_ddt_int              : partition(disjoint, rgn_cvars_ddt, ispace(int1d)),
     --
     part_cvars_now_int              : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_all              : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_i_prev_send      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_i_next_send      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_j_prev_send      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_j_next_send      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_i_prev_recv      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_i_next_recv      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_j_prev_recv      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     part_cvars_now_j_next_recv      : partition(disjoint, rgn_cvars_now, ispace(int1d)),
     --
     part_grad_vel_coll_int          : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_all          : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_i_prev_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_i_next_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_j_prev_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_j_next_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_i_prev_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_i_next_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_j_prev_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     part_grad_vel_coll_j_next_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
     --
     part_grid                       : partition(disjoint, rgn_grid, ispace(int1d)),
     part_meta                       : partition(disjoint, rgn_meta, ispace(int1d))
)
where
    writes(rgn_cvars_ddt),
    reads(rgn_grid, rgn_meta),
    reads writes(rgn_cvars_now, rgn_grad_vel_coll)
do
    [grid.fillGhostsLevel(CVARS)](
        level,
        rgn_meta, part_meta, rgn_cvars_now,
        part_cvars_now_i_prev_send,
        part_cvars_now_i_next_send,
        part_cvars_now_j_prev_send,
        part_cvars_now_j_next_send,
        part_cvars_now_i_prev_recv,
        part_cvars_now_i_next_recv,
        part_cvars_now_j_prev_recv,
        part_cvars_now_j_next_recv
    );
    for color in part_meta.colors do
        var pid = int1d(color);
        if (part_meta[color][color].level == level) then
            solver.calcGradVelColl(part_grad_vel_coll_int[pid], part_cvars_now_all[pid], part_meta[pid]);
        end
    end
    [grid.fillGhostsLevel(GRAD_VEL)](
        level,
        rgn_meta, part_meta, rgn_grad_vel_coll,
        part_grad_vel_coll_i_prev_send,
        part_grad_vel_coll_i_next_send,
        part_grad_vel_coll_j_prev_send,
        part_grad_vel_coll_j_next_send,
        part_grad_vel_coll_i_prev_recv,
        part_grad_vel_coll_i_next_recv,
        part_grad_vel_coll_j_prev_recv,
        part_grad_vel_coll_j_next_recv
    );
    -- fill(rgn_cvars_ddt.{mass, mmtx, mmty, enrg}, 0.0);
    for color in part_meta.colors do
        var pid = int1d(color);
        if (part_meta[color][color].level == level) then
            solver.calcRHSLeaf(part_cvars_ddt_int[pid], part_cvars_now_all[pid], part_grad_vel_coll_all[pid], part_grid[pid], part_meta[pid]);
        end
    end
end



-- Conduct RK3 update
task solver.SSPRK3Launch(
    level                           : int,
    dt                              : double,
    rgn_cvars_0                     : region(ispace(int3d), CVARS   ),
    rgn_cvars_1                     : region(ispace(int3d), CVARS   ),
    rgn_cvars_2                     : region(ispace(int3d), CVARS   ),
    rgn_grad_vel_coll               : region(ispace(int3d), GRAD_VEL),
    rgn_grid                        : region(ispace(int3d), grid_fsp),    
    rgn_meta                        : region(ispace(int1d), grid_meta_fsp),
    --
    part_cvars_0_int                : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_all                : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_i_prev_send        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_i_next_send        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_j_prev_send        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_j_next_send        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_i_prev_recv        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_i_next_recv        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_j_prev_recv        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    part_cvars_0_j_next_recv        : partition(disjoint, rgn_cvars_0, ispace(int1d)),
    --
    part_cvars_1_int                : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_all                : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_i_prev_send        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_i_next_send        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_j_prev_send        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_j_next_send        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_i_prev_recv        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_i_next_recv        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_j_prev_recv        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    part_cvars_1_j_next_recv        : partition(disjoint, rgn_cvars_1, ispace(int1d)),
    --
    part_cvars_2_int                : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_all                : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_i_prev_send        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_i_next_send        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_j_prev_send        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_j_next_send        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_i_prev_recv        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_i_next_recv        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_j_prev_recv        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    part_cvars_2_j_next_recv        : partition(disjoint, rgn_cvars_2, ispace(int1d)),
    --
    part_grad_vel_coll_int          : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_all          : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_i_prev_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_i_next_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_j_prev_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_j_next_send  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_i_prev_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_i_next_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_j_prev_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    part_grad_vel_coll_j_next_recv  : partition(disjoint, rgn_grad_vel_coll, ispace(int1d)),
    --
    part_grid                       : partition(disjoint, rgn_grid, ispace(int1d)),
    part_meta                       : partition(disjoint, rgn_meta, ispace(int1d))
)
where
    reads writes(rgn_cvars_0, rgn_cvars_1, rgn_cvars_2, rgn_grad_vel_coll),
    reads(rgn_grid, rgn_meta)
do

    -- Stage 0
    solver.calcRHSLaunch(
        level,
        rgn_cvars_1, rgn_cvars_0, rgn_grad_vel_coll, rgn_grid, rgn_meta, part_cvars_1_int,
        part_cvars_0_int,
        part_cvars_0_all,
        part_cvars_0_i_prev_send,
        part_cvars_0_i_next_send,
        part_cvars_0_j_prev_send,
        part_cvars_0_j_next_send,
        part_cvars_0_i_prev_recv,
        part_cvars_0_i_next_recv,
        part_cvars_0_j_prev_recv,
        part_cvars_0_j_next_recv,
        part_grad_vel_coll_int,
        part_grad_vel_coll_all,
        part_grad_vel_coll_i_prev_send,
        part_grad_vel_coll_i_next_send,
        part_grad_vel_coll_j_prev_send,
        part_grad_vel_coll_j_next_send,
        part_grad_vel_coll_i_prev_recv,
        part_grad_vel_coll_i_next_recv,
        part_grad_vel_coll_j_prev_recv,
        part_grad_vel_coll_j_next_recv,
        part_grid, part_meta
    )
    for color in part_meta.colors do
        var pid = int1d(color);
        if (part_meta[color][color].level == level) then
            [SSPRK3Stage(0)](dt, part_cvars_0_int[pid], part_cvars_1_int[pid]);
        end
    end

    -- Stage 1
    solver.calcRHSLaunch(
        level,
        rgn_cvars_2, rgn_cvars_1, rgn_grad_vel_coll, rgn_grid, rgn_meta, part_cvars_2_int,
        part_cvars_1_int,
        part_cvars_1_all,
        part_cvars_1_i_prev_send,
        part_cvars_1_i_next_send,
        part_cvars_1_j_prev_send,
        part_cvars_1_j_next_send,
        part_cvars_1_i_prev_recv,
        part_cvars_1_i_next_recv,
        part_cvars_1_j_prev_recv,
        part_cvars_1_j_next_recv,
        part_grad_vel_coll_int,
        part_grad_vel_coll_all,
        part_grad_vel_coll_i_prev_send,
        part_grad_vel_coll_i_next_send,
        part_grad_vel_coll_j_prev_send,
        part_grad_vel_coll_j_next_send,
        part_grad_vel_coll_i_prev_recv,
        part_grad_vel_coll_i_next_recv,
        part_grad_vel_coll_j_prev_recv,
        part_grad_vel_coll_j_next_recv,
        part_grid, part_meta
    )
    for color in part_meta.colors do
        var pid = int1d(color);
        if (part_meta[color][color].level == level) then
            [SSPRK3Stage(1)](dt, part_cvars_0_int[pid], part_cvars_1_int[pid], part_cvars_2_int[pid]);
        end
    end

    -- Stage 2
    solver.calcRHSLaunch(
        level,
        rgn_cvars_1, rgn_cvars_2, rgn_grad_vel_coll, rgn_grid, rgn_meta, part_cvars_1_int,
        part_cvars_2_int,
        part_cvars_2_all,
        part_cvars_2_i_prev_send,
        part_cvars_2_i_next_send,
        part_cvars_2_j_prev_send,
        part_cvars_2_j_next_send,
        part_cvars_2_i_prev_recv,
        part_cvars_2_i_next_recv,
        part_cvars_2_j_prev_recv,
        part_cvars_2_j_next_recv,
        part_grad_vel_coll_int,
        part_grad_vel_coll_all,
        part_grad_vel_coll_i_prev_send,
        part_grad_vel_coll_i_next_send,
        part_grad_vel_coll_j_prev_send,
        part_grad_vel_coll_j_next_send,
        part_grad_vel_coll_i_prev_recv,
        part_grad_vel_coll_i_next_recv,
        part_grad_vel_coll_j_prev_recv,
        part_grad_vel_coll_j_next_recv,
        part_grid, part_meta
    )
    for color in part_meta.colors do
        var pid = int1d(color);
        if (part_meta[color][color].level == level) then
            [SSPRK3Stage(2)](dt, part_cvars_0_int[pid], part_cvars_1_int[pid], part_cvars_2_int[pid]);
        end
    end
end

local __demand(__inline) task checkNaN(
    tag     : int,
    patches : region(ispace(int3d), double)
)
where
    reads (patches)
do
    -- format.println("invoking tag = {}", tag);
    for cij in patches.ispace do
        if (cmath.isnan(patches[cij]) ~= 0) then
            -- format.println("tag = {}", tag);
            regentlib.assert(false, "NaN detected");
        end
    end
end

local __demand(__inline) task checkZero(
    tag     : int,
    patches : region(ispace(int3d), double)
)
where
    reads (patches)
do
    -- format.println("invoking tag = {}", tag);
    for cij in patches.ispace do
        if (patches[cij]*patches[cij] < 1e-32) then
            format.println("tag = {}", tag);
            break
            -- regentlib.assert(false, "Zero detected");
        end
    end
end



local task dumpDensity(
    fname          :  rawstring,
    level          :  int,
    rgn_cvars      :  region(ispace(int3d), CVARS),
    rgn_grid       :  region(ispace(int3d), grid_fsp),    
    rgn_meta       :  region(ispace(int1d), grid_meta_fsp),
    patches_cvars  :  partition(disjoint, rgn_cvars, ispace(int1d)),
    patches_grid   :  partition(disjoint, rgn_grid, ispace(int1d))
)
where
    reads (rgn_cvars, rgn_grid.{x, y}, rgn_meta.{level,i_coord,j_coord})
do
    var file = c.fopen(fname, "w")
    --c.fprintf(file, "%8s, %8s, %8s, %8s, %8s, %23s, %23s, %23s\n", "color_id", "patch_i", "patch_j", "local_i", "local_j", "x", "y", "density")
    c.fprintf(file, "%8s, %8s, %8s, %8s, %8s, %8s, %23s, %23s, %23s, %23s, %23s, %23s\n", "color_id", "level", "patch_i", "patch_j", "local_i", "local_j", "x", "y", "vel-x", "vel-y", "temperature", "pressure")
    for pid in rgn_meta.ispace do
        if rgn_meta[pid].level <= level and rgn_meta[pid].level > -1 then
            var cvars_patch = patches_cvars[pid]
            var grid_patch  = patches_grid[pid]
            var level_ij    = rgn_meta[pid].level
            var patch_i     = rgn_meta[pid].i_coord
            var patch_j     = rgn_meta[pid].j_coord
            for cij in cvars_patch.ispace do
                -- var density = cvars_patch[cij].mass
                var pvars = conservativeToPrimitive(cvars_patch[cij].mass, cvars_patch[cij].mmtx, cvars_patch[cij].mmty, cvars_patch[cij].enrg)
                var x = grid_patch[cij].x
                var y = grid_patch[cij].y
                c.fprintf(file, "%8d, %8d, %8d, %8d, %8d, %8d, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e\n",
                                int(pid), level_ij, patch_i, patch_j, cij.y, cij.z, x, y, pvars.u, pvars.v, pvars.T, pvars.p)
            end
        end
    end
    c.fclose(file)
end

__demand(__inline)
task solver.setRefineFlagsLeaf(
    rgn_patch_meta  : region(ispace(int1d), grid_meta_fsp),
    rgn_patch_cvars : region(ispace(int3d),         CVARS)
)
where
    reads (rgn_patch_cvars),
    reads writes (rgn_patch_meta)
do
    var isp = rgn_patch_cvars.ispace
    var pid = isp.bounds.lo.x
    var T0  = 1.0 / (eos.Rg * eos.gamma)
    var U0  = 0.5 * cmath.sqrt(eos.gamma * eos.Rg * T0)


    var threshold : double = 0.98;
    -- if (rgn_patch_meta[int1d(pid)].level == 0) then
    --     threshold = 0.98
    -- elseif (rgn_patch_meta[int1d(pid)].level == 1) then
    --     threshold = 0.0 -- 0.88
    -- end

    -- Set refine flag
    var count : int = 0;
    for ij in isp do
        count += int(rgn_patch_cvars[ij].mass < threshold)
    end
    if (count > 2) then
        rgn_patch_meta[int1d(pid)].refine_req = true;
    else
        rgn_patch_meta[int1d(pid)].refine_req = false;
    end
end

__demand(__inline)
task solver.setCoarsenFlagsLeaf(
    rgn_patch_meta  : region(ispace(int1d), grid_meta_fsp),
    rgn_patch_cvars : region(ispace(int3d),         CVARS)
)
where
    reads (rgn_patch_cvars),
    reads writes (rgn_patch_meta)
do
    var isp = rgn_patch_cvars.ispace
    var pid = isp.bounds.lo.x
    var T0  = 1.0 / (eos.Rg * eos.gamma)
    var U0  = 0.5 * cmath.sqrt(eos.gamma * eos.Rg * T0)

    var threshold : double = 0.99;
    -- if (rgn_patch_meta[int1d(pid)].level == 0) then
    --     threshold = 0.99
    -- else
    --     threshold = 9999.9-- 0.88
    -- end

    -- Set coarsen flag
    var count = 0
    for ij in isp do
        count += int(rgn_patch_cvars[ij].mass > threshold)
    end
    if (count > 254) then
        --rgn_patch_meta[int1d(pid)].coarsen_req = true;
        if (rgn_patch_meta[int1d(pid)].child[0] > -1) then
            rgn_patch_meta[int1d(rgn_patch_meta[int1d(pid)].child[0])].coarsen_req = true
            rgn_patch_meta[int1d(rgn_patch_meta[int1d(pid)].child[1])].coarsen_req = true
            rgn_patch_meta[int1d(rgn_patch_meta[int1d(pid)].child[2])].coarsen_req = true
            rgn_patch_meta[int1d(rgn_patch_meta[int1d(pid)].child[3])].coarsen_req = true
        end
    end
end


task solver.adjustMesh(
    rgn_patches_meta  : region(ispace(int1d), grid_meta_fsp),
    rgn_patches_grid  : region(ispace(int3d),      grid_fsp),
    rgn_patches_cvars : region(ispace(int3d),         CVARS),
    patches_meta      : partition(disjoint, rgn_patches_meta , ispace(int1d)),
    patches_grid_int  : partition(disjoint, rgn_patches_grid , ispace(int1d)),
    patches_cvars_int : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars     : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    -------------------------------
    patches_cvars_i_prev_send : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_i_next_send : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_j_prev_send : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_j_next_send : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_i_prev_recv : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_i_next_recv : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_j_prev_recv : partition(disjoint, rgn_patches_cvars, ispace(int1d)),
    patches_cvars_j_next_recv : partition(disjoint, rgn_patches_cvars, ispace(int1d))
)
where
    reads writes (rgn_patches_meta, rgn_patches_grid, rgn_patches_cvars)
do 
    for l = 0, grid.level_max do
        for color in patches_meta.colors do
            if (patches_meta[int1d(color)][int1d(color)].level == l) then
                solver.setRefineFlagsLeaf(patches_meta[int1d(color)], patches_cvars_int[int1d(color)])
            end
        end
        grid.refineInit(rgn_patches_meta, patches_meta)
        if (l > 0) then
            [grid.fillGhostsLevel(CVARS)](
                l - 1,
                rgn_patches_meta, patches_meta, rgn_patches_cvars,
                patches_cvars_i_prev_send,
                patches_cvars_i_next_send,
                patches_cvars_j_prev_send,
                patches_cvars_j_next_send,
                patches_cvars_i_prev_recv,
                patches_cvars_i_next_recv,
                patches_cvars_j_prev_recv,
                patches_cvars_j_next_recv
            );
        end
        for color in patches_meta.colors do
            var parent_meta = patches_meta[int1d(color)][int1d(color)]
            if (parent_meta.level > -1 and parent_meta.level == l - 1 and parent_meta.refine_req) then
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[0])], patches_meta[int1d(parent_meta.child[0])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[1])], patches_meta[int1d(parent_meta.child[1])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[2])], patches_meta[int1d(parent_meta.child[2])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[3])], patches_meta[int1d(parent_meta.child[3])])
                grid.upsample (
                    patches_cvars[int1d(color)].{mass},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mass}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mmtx}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mmty}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[0])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[1])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[2])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[3])].{enrg}
                )
            end
        end
        [grid.fillGhostsLevel(CVARS)](
            l,
            rgn_patches_meta, patches_meta, rgn_patches_cvars,
            patches_cvars_i_prev_send,
            patches_cvars_i_next_send,
            patches_cvars_j_prev_send,
            patches_cvars_j_next_send,
            patches_cvars_i_prev_recv,
            patches_cvars_i_next_recv,
            patches_cvars_j_prev_recv,
            patches_cvars_j_next_recv
        );
        for color in patches_meta.colors do
            var parent_meta = patches_meta[int1d(color)][int1d(color)]
            if (parent_meta.level == l and parent_meta.child[0] > -1) then
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[0])], patches_meta[int1d(parent_meta.child[0])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[1])], patches_meta[int1d(parent_meta.child[1])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[2])], patches_meta[int1d(parent_meta.child[2])])
                solver.setGridPointCoordinates(patches_grid_int[int1d(parent_meta.child[3])], patches_meta[int1d(parent_meta.child[3])])
                grid.upsample (
                    patches_cvars[int1d(color)].{mass},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mass},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mass}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mmtx},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mmtx}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[0])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[1])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[2])].{mmty},
                    patches_cvars_int[int1d(parent_meta.child[3])].{mmty}
                )
                grid.upsample (
                    patches_cvars[int1d(color)].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[0])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[1])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[2])].{enrg},
                    patches_cvars_int[int1d(parent_meta.child[3])].{enrg}
                )
            end
        end
        grid.refineEnd(rgn_patches_meta);
    end

    -- for l = grid.level_max, 0, -1 do
    --     for color in patches_meta.colors do
    --         if (patches_meta[int1d(color)][int1d(color)].level == l) then
    --             solver.setCoarsenFlagsLeaf(patches_meta[int1d(color)], patches_cvars_int[int1d(color)])
    --         end
    --     end
    --     grid.coarsenInit(rgn_patches_meta, patches_meta)
    --     grid.coarsenEnd(rgn_patches_meta, patches_meta)
    -- end
end




local task writeActiveMeta(
    fname               : rawstring,
    meta_patches_region : region(ispace(int1d), grid_meta_fsp),
    meta_patches        : partition(disjoint, complete, meta_patches_region, ispace(int1d))
)
where
    reads (meta_patches_region)
do
    var file = c.fopen(fname, "w")
    c.fprintf(file, "%7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s, %7s\n",
                "pid", "level", "i_coord", "j_coord", "i_prev", "i_next", "j_prev", "j_next", "parent", "child0", "child1", "child2", "child3", "r_req", "c_req");
    for pid in meta_patches.colors do
        var patch = meta_patches[pid][pid]
        if (patch.level > -1) then
            c.fprintf(file, "%7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d, %7d\n",
                int(pid), patch.level, patch.i_coord, patch.j_coord,
                patch.i_prev, patch.i_next, patch.j_prev, patch.j_next,
                patch.parent, patch.child[0], patch.child[1], patch.child[2], patch.child[3],
                patch.refine_req, patch.coarsen_req);
        end
    end -- for pid
    c.fclose(file)
end -- task


task solver.main()

    var args = c.legion_runtime_get_input_args()
    var loop_cnt:     int = c.atoi(args.argv[1]);
    var time_step: double = c.atof(args.argv[2]);
    var stride:       int = c.atoi(args.argv[3]);
    -- 
    c.printf("Solver initialization:")
    c.printf("  -- Patch size (interior) : %d x %d\n", grid.patch_size, grid.patch_size)
    c.printf("  -- Ghost points on each side in each dimension: %d\n", grid.num_ghosts)
    c.printf("  -- Base level patches: %d x %d\n", grid.num_base_patches_i, grid.num_base_patches_j)
    c.printf("  -- Maximum level of refinement: %d\n", grid.level_max)
    c.printf("  -- Maximum possible allocated patches: %d\n", grid.num_patches_max)

    -- ispace(int1d, grid.num_patches_max, 0)
    var color_space  = [grid.createColorSpace()];

    -- region(ispace(int3d, {grid.num_patches_max, grid.full_patch_size, grid.full_patch_size}, {0, grid.idx_min, grid.idx_min}), grid_fsp)
    var rgn_patches_grid     = [grid.createDataRegion(grid_fsp)];
    var rgn_patches_grad_vel = [grid.createDataRegion(GRAD_VEL)];
    var rgn_patches_cvars_0  = [grid.createDataRegion(CVARS)];
    var rgn_patches_cvars_1  = [grid.createDataRegion(CVARS)];
    var rgn_patches_cvars_2  = [grid.createDataRegion(CVARS)];

    -- region(ispace(int1d, grid.num_patches_max, 0), grid_meta_fsp)
    var rgn_patches_meta = [grid.createMetaRegion()];


    -- CREATE PARTITIONS
    var patches_meta             = grid.createPartitionOfMetaPatches(rgn_patches_meta); -- complete partition of rgn_patches_meta
    var patches_grid             = [grid.createPartitionOfFullPatches     (grid_fsp)](rgn_patches_grid); -- complete partition of rgn_patches_grid
    var patches_grid_int         = [grid.createPartitionOfInteriorPatches (grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_send = [grid.createPartitionOfINextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_i_next_recv = [grid.createPartitionOfINextRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_send = [grid.createPartitionOfJNextSendBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(grid_fsp)](rgn_patches_grid);
    var patches_grid_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(grid_fsp)](rgn_patches_grid);

    var patches_grad_vel             = [grid.createPartitionOfFullPatches     (GRAD_VEL)](rgn_patches_grad_vel); -- complete partition of rgn_patches_grad_vel
    var patches_grad_vel_int         = [grid.createPartitionOfInteriorPatches (GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_next_send = [grid.createPartitionOfINextSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_i_next_recv = [grid.createPartitionOfINextRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_next_send = [grid.createPartitionOfJNextSendBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);
    var patches_grad_vel_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(GRAD_VEL)](rgn_patches_grad_vel);

    var patches_cvars_0             = [grid.createPartitionOfFullPatches     (CVARS)](rgn_patches_cvars_0); -- complete partition of rgn_patches_cvars_0
    var patches_cvars_0_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_0);
    var patches_cvars_0_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_0);

    var patches_cvars_1             = [grid.createPartitionOfFullPatches     (CVARS)](rgn_patches_cvars_1); -- complete partition of rgn_patches_cvars_1
    var patches_cvars_1_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_1);
    var patches_cvars_1_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_1);

    var patches_cvars_2             = [grid.createPartitionOfFullPatches(CVARS)](rgn_patches_cvars_2); -- complete partition of rgn_patches_cvars_2
    var patches_cvars_2_int         = [grid.createPartitionOfInteriorPatches (CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_prev_send = [grid.createPartitionOfIPrevSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_next_send = [grid.createPartitionOfINextSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_prev_recv = [grid.createPartitionOfIPrevRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_i_next_recv = [grid.createPartitionOfINextRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_prev_send = [grid.createPartitionOfJPrevSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_next_send = [grid.createPartitionOfJNextSendBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_prev_recv = [grid.createPartitionOfJPrevRecvBuffers(CVARS)](rgn_patches_cvars_2);
    var patches_cvars_2_j_next_recv = [grid.createPartitionOfJNextRecvBuffers(CVARS)](rgn_patches_cvars_2);


    -- INITIALIZE DATA PATCHES
    fill(rgn_patches_grid.{x, y}, 0.0);

    -- INITIALIZE META PATCHES to avoid warnings (write_discard not supported)
    fill(rgn_patches_meta.{level, i_coord, j_coord, i_prev, i_next, j_prev, j_next, parent}, 0);
    fill(rgn_patches_meta.child, [terralib.constant(`arrayof(int, 0, 0, 0, 0))]);
    fill(rgn_patches_meta.{refine_req, coarsen_req}, false);
    grid.metaGridInit(rgn_patches_meta, patches_meta);

    fill(rgn_patches_cvars_0.{mass, mmtx, mmty, enrg}, 0.0);
    fill(rgn_patches_cvars_1.{mass, mmtx, mmty, enrg}, 0.0);
    fill(rgn_patches_cvars_2.{mass, mmtx, mmty, enrg}, 0.0);

    __demand(__index_launch)
    for color = 0, num_base_patches_i * num_base_patches_j do
       solver.setGridPointCoordinates(patches_grid_int[color], patches_meta[color])
    end

    -- INITIALIZE GRAD_VEL to get rid of warnings (write_discard not supported)
    fill(rgn_patches_grad_vel.{dudx, dudy, dvdx, dvdy}, 0.0);

    __demand(__index_launch)
    for color = 0, num_base_patches_i * num_base_patches_j do
        problem_config.setInitialCondition(patches_grid_int[color], patches_meta[color], patches_cvars_0_int[color])
    end

    -- TODO: Refine mesh based on the initial profile
    -- TODO: Set coordinate on refined mesh
    -- TODO: Redo setInitialCondition on refined mesh

    solver.adjustMesh(rgn_patches_meta, rgn_patches_grid, rgn_patches_cvars_0, patches_meta, patches_grid_int, patches_cvars_0_int, patches_cvars_0,
                      patches_cvars_0_i_prev_send, patches_cvars_0_i_next_send, patches_cvars_0_j_prev_send, patches_cvars_0_j_next_send,
                      patches_cvars_0_i_prev_recv, patches_cvars_0_i_next_recv, patches_cvars_0_j_prev_recv, patches_cvars_0_j_next_recv);
    dumpDensity("density_000000.dat", grid.level_max, rgn_patches_cvars_0, rgn_patches_grid, rgn_patches_meta, patches_cvars_0_int, patches_grid_int);
    writeActiveMeta("mesh_000000.dat", rgn_patches_meta, patches_meta);
    --
    --
    -- TODO: Recursively refine mesh to higher levels
    --
    --
    -- for pid in patches_meta.colors do
    --     if ((patches_meta[pid][pid].level > (-1))) then
    --         problem_config.setInitialCondition(patches_grid[int1d(pid)], patches_meta[int1d(pid)], patches_cvars_0[int1d(pid)]);
    --     end
    -- end


    -- for pid in patches_meta.colors do
    --     if ((patches_meta[pid][pid].level > (-1))) then
    --         solver.calcGradVelColl(patches_grad_vel_int[int1d(pid)], patches_cvars_0[int1d(pid)], patches_meta[int1d(pid)]);
    --         solver.calcRHSLeaf(patches_cvars_1_int[int1d(pid)], patches_cvars_0[int1d(pid)], patches_grad_vel[int1d(pid)], patches_grid[int1d(pid)], patches_meta[int1d(pid)]);
    --     end
    -- end
    for i = 0, loop_cnt do
        solver.SSPRK3Launch(
            0,
            time_step,
            rgn_patches_cvars_0,
            rgn_patches_cvars_1,
            rgn_patches_cvars_2,
            rgn_patches_grad_vel,
            rgn_patches_grid,
            rgn_patches_meta,
            -- 
            patches_cvars_0_int,
            patches_cvars_0,
            patches_cvars_0_i_prev_send,
            patches_cvars_0_i_next_send,
            patches_cvars_0_j_prev_send,
            patches_cvars_0_j_next_send,
            patches_cvars_0_i_prev_recv,
            patches_cvars_0_i_next_recv,
            patches_cvars_0_j_prev_recv,
            patches_cvars_0_j_next_recv,
            --
            patches_cvars_1_int,
            patches_cvars_1,
            patches_cvars_1_i_prev_send,
            patches_cvars_1_i_next_send,
            patches_cvars_1_j_prev_send,
            patches_cvars_1_j_next_send,
            patches_cvars_1_i_prev_recv,
            patches_cvars_1_i_next_recv,
            patches_cvars_1_j_prev_recv,
            patches_cvars_1_j_next_recv,
            --
            patches_cvars_2_int,
            patches_cvars_2,
            patches_cvars_2_i_prev_send,
            patches_cvars_2_i_next_send,
            patches_cvars_2_j_prev_send,
            patches_cvars_2_j_next_send,
            patches_cvars_2_i_prev_recv,
            patches_cvars_2_i_next_recv,
            patches_cvars_2_j_prev_recv,
            patches_cvars_2_j_next_recv,
            --
            patches_grad_vel_int,
            patches_grad_vel,
            patches_grad_vel_i_prev_send,
            patches_grad_vel_i_next_send,
            patches_grad_vel_j_prev_send,
            patches_grad_vel_j_next_send,
            patches_grad_vel_i_prev_recv,
            patches_grad_vel_i_next_recv,
            patches_grad_vel_j_prev_recv,
            patches_grad_vel_j_next_recv,
            --
            patches_grid,
            patches_meta
        );

        if i % stride == 0 then
            var filename_dat : &int8 = [&int8] (c.malloc(64*8))
            var filename_msh : &int8 = [&int8] (c.malloc(64*8))
            c.sprintf(filename_dat, "density_%06d.dat", i+1);
            c.sprintf(filename_msh, "mesh_%06d.dat", i+1);
            dumpDensity(filename_dat, grid.level_max, rgn_patches_cvars_0, rgn_patches_grid, rgn_patches_meta, patches_cvars_0_int, patches_grid_int);
            writeActiveMeta(filename_msh, rgn_patches_meta, patches_meta);
        end
        -- c.free(filename); -- should not free until dumpDensity finishes
        solver.adjustMesh(rgn_patches_meta, rgn_patches_grid, rgn_patches_cvars_0, patches_meta, patches_grid_int, patches_cvars_0_int, patches_cvars_0,
                          patches_cvars_0_i_prev_send, patches_cvars_0_i_next_send, patches_cvars_0_j_prev_send, patches_cvars_0_j_next_send,
                          patches_cvars_0_i_prev_recv, patches_cvars_0_i_next_recv, patches_cvars_0_j_prev_recv, patches_cvars_0_j_next_recv);
    end

end


return solver
