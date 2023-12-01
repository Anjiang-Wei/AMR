import "regent"

local config = {}

-- INPUTS FOR AMR BASICS --
config.num_base_patches_i =  8; -- number of base-level patches in i-direction
config.num_base_patches_j =  8; -- number of base-level patches in j-direction
config.patch_size         = 16; -- number of grid points within each patch in each direction
config.level_max          =  2; -- maximum allowable refinement level (inclusive) where base level is 0
config.num_ghosts         =  4; -- number of ghost grid points in each patch in each direction on each side


-- BASIC NUMERICS --
config.numerics_modules = "numerics"


-- INPUTS FOR NAVIER-STOKES SOLVER MESH GENERATION --
config.domain_length_x =  1.0 -- physical domain length in x-direction
config.domain_length_y =  1.0 -- physical domain length in y-direction
config.domain_shift_x  = -0.5 -- physical domain coordinate shift in x-direction (if no shift, the domain origin is at the lower-left corner of the domain)
config.domain_shift_y  = -0.5 -- physical domain coordinate shift in y-direction (if no shift, the domain origin is at the lower-left corner of the domain)


return config
