import "regent"

fspace CVARS {
    mass : double,
    mmtx : double,
    mmty : double,
    enrg : double
}


-- fspace PVARS {
--     rho : double,
--     u   : double,
--     v   : double,
--     T   : double,
--     p   : double
-- }

fspace GRAD_VEL {
    dudx : double,
    dudy : double,
    dvdx : double,
    dvdy : double
}