import "regent"

local config = {}

config.eos       = {}
config.transport = {}

-- INPUTS FOR AMR BASICS --
config.num_base_patches_i =  6; -- number of base-level patches in i-direction
config.num_base_patches_j =  6; -- number of base-level patches in j-direction
config.patch_size         = 16; -- number of grid points within each patch in each direction
config.level_max          =  1; -- maximum allowable refinement level (inclusive) where base level is 0
config.num_ghosts         =  4; -- number of ghost grid points in each patch in each direction on each side


-- BASIC NUMERICS --
config.numerics_modules = "numerics"


-- INPUTS FOR NAVIER-STOKES SOLVER MESH GENERATION --
config.domain_length_x = 12.0 -- physical domain length in x-direction
config.domain_length_y = 12.0 -- physical domain length in y-direction
config.domain_shift_x  = -6.0 -- physical domain coordinate shift in x-direction (if no shift, the domain origin is at the lower-left corner of the domain)
config.domain_shift_y  = -6.0 -- physical domain coordinate shift in y-direction (if no shift, the domain origin is at the lower-left corner of the domain)



-- EQUATION OF STATE
config.eos.Rg    = 1.0    -- specific gas constant
config.eos.gamma = 1.4    -- ratio of of specific heats


-- TRANSPORT PROPERTIES
config.transport.T_ref    = 1.0      -- reference temperature for calculation of viscosity using power-law
config.transport.mu_ref   = 0.0 -- 1.0e-3   -- reference viscosity for calculation of viscosity using power-law
config.transport.visc_exp = 0.76     -- exponent of temperature dependent viscosity
config.transport.Pr       = 0.7      -- Prandtl number


return config
