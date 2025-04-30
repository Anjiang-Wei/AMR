import matplotlib
matplotlib.use("agg")
import os
import cupy as cp
import numpy as np  # only for host‑side utilities (plotting, saving)
import matplotlib.pyplot as plt
import h5py

"""====================================================================
CuPy implementation of the 2‑D implosion test originally written with
NumPy + Numba.  The goal is **functional equivalence**: given the same
resolution and end‑time, the HDF5 snapshots should match those produced
by the CPU version to within round‑off error (\u2248 1e‑12).

Performance has NOT been tuned – many kernels are still expressed with
Python loops, so expect modest speed‑ups or even slow‑downs for large N.
This code is meant as a correctness baseline that keeps every array on
the GPU; you can later optimise the innermost loops with CuPy RawKernel
or Numba‑CUDA.
===================================================================="""

# ------------------------- simulation parameters -------------------------

t_final      = 1.0   # physical end‑time
CFL          = 0.10  # CFL number
dt_output    = 1e‑3  # snapshot interval
Rg           = 1.0   # gas constant
gamma        = 1.4   # heat‑capacity ratio
L            = 0.6   # domain half‑width (square domain → size L×L)
N            = 1024  # grid points per dimension

fig_name_fmt = "./ImplosionFigs2/vis_implosion_{:06d}.pdf"
dat_name_fmt = "./ImplosionData2/dat_implosion_{:06d}.h5"

# --------------------------- helper containers ---------------------------

class ConsVars:
    """Conservative variables wrapper living entirely on the GPU."""

    def __init__(self, N: int):
        self.N = N
        self.mass = cp.empty((N, N), dtype=cp.float64)
        self.mmtx = cp.empty((N, N), dtype=cp.float64)
        self.mmty = cp.empty((N, N), dtype=cp.float64)
        self.enrg = cp.empty((N, N), dtype=cp.float64)

    # convenience accessors ------------------------------------------------
    def get(self):
        return [self.mass, self.mmtx, self.mmty, self.enrg]

    def convertToPrim(self):
        """Return (rho, u, v, T, p) on the GPU."""
        rho = cp.copy(self.mass)
        u   = self.mmtx / self.mass
        v   = self.mmty / self.mass
        T   = (self.enrg / self.mass - 0.5 * (u * u + v * v)) * (gamma - 1.0) / Rg
        p   = rho * Rg * T
        return rho, u, v, T, p

# --------------------------- primitive ↔︎ conservative -------------------

def conservativeToPrimitive(mass, mmtx, mmty, enrg):
    rho = cp.copy(mass)
    u   = mmtx / mass
    v   = mmty / mass
    T   = (enrg / mass - 0.5 * (u * u + v * v)) * (gamma - 1.0) / Rg
    p   = rho * Rg * T
    return rho, u, v, T, p

# ----------------------------- Roe averages -----------------------------

def roeAvg(rhoL, rhoR, uL, uR, vL, vR, TL, TR):
    rhoL_sqrt = cp.sqrt(rhoL)
    rhoR_sqrt = cp.sqrt(rhoR)
    rho_fact  = 1.0 / (rhoL_sqrt + rhoR_sqrt)

    cv  = Rg / (gamma - 1.0)
    qL  = uL * uL + vL * vL
    qR  = uR * uR + vR * vR
    eL  = cv * TL + 0.5 * qL
    eR  = cv * TR + 0.5 * qR
    hL  = eL + Rg * TL
    hR  = eR + Rg * TR

    rho_avg = rhoL_sqrt * rhoR_sqrt
    u_avg   = (rhoL_sqrt * uL + rhoR_sqrt * uR) * rho_fact
    v_avg   = (rhoL_sqrt * vL + rhoR_sqrt * vR) * rho_fact
    e_avg   = (rhoL_sqrt * eL + rhoR_sqrt * eR) * rho_fact
    h_avg   = (rhoL_sqrt * hL + rhoR_sqrt * hR) * rho_fact
    c_avg   = cp.sqrt((gamma - 1.0) * (h_avg - 0.5 * (u_avg * u_avg + v_avg * v_avg)))

    return u_avg, v_avg, c_avg, h_avg

# ------------------------- characteristic transforms --------------------

def conservativeToCharacteristic(U0, U1, U2, U3, c_avg, u_avg, v_avg, g, l):
    ek_avg   = 0.5 * (u_avg * u_avg + v_avg * v_avg)
    gm1byc2  = 0.5 * (gamma - 1.0) / (c_avg * c_avg)
    Ug       = u_avg * g[0] + v_avg * g[1]
    Ul       = u_avg * l[0] + v_avg * l[1]

    ch0 = U0 * (gm1byc2 * ek_avg + 0.5 * Ug / c_avg) \
        - U1 * (0.5 * g[0] / c_avg + gm1byc2 * u_avg) \
        - U2 * (0.5 * g[1] / c_avg + gm1byc2 * v_avg) \
        + U3 * gm1byc2

    ch1 = -U0 * Ul + U1 * l[0] + U2 * l[1]

    ch2 = U0 * (1.0 - 2.0 * gm1byc2 * ek_avg) \
        + U1 * (2.0 * gm1byc2 * u_avg) \
        + U2 * (2.0 * gm1byc2 * v_avg) \
        - 2.0 * U3 * gm1byc2

    ch3 = U0 * (gm1byc2 * ek_avg - 0.5 * Ug / c_avg) \
        + U1 * (0.5 * g[0] / c_avg - gm1byc2 * u_avg) \
        + U2 * (0.5 * g[1] / c_avg - gm1byc2 * v_avg) \
        + U3 * gm1byc2

    return ch0, ch1, ch2, ch3


def characteristicToConservative(ch0, ch1, ch2, ch3, c_avg, u_avg, v_avg, h_avg, g, l):
    Ug      = u_avg * g[0] + v_avg * g[1]
    Ul      = u_avg * l[0] + v_avg * l[1]
    ek_avg  = 0.5 * (u_avg * u_avg + v_avg * v_avg)

    U0 = ch0 + ch2 + ch3

    U1 = ch0 * (u_avg - c_avg * g[0]) + ch1 * l[0] + ch2 * u_avg + ch3 * (u_avg + c_avg * g[0])

    U2 = ch0 * (v_avg - c_avg * g[1]) + ch1 * l[1] + ch2 * v_avg + ch3 * (v_avg + c_avg * g[1])

    U3 = ch0 * (h_avg - c_avg * Ug) + ch1 * Ul + ch2 * ek_avg + ch3 * (h_avg + c_avg * Ug)

    return U0, U1, U2, U3

# ------------------------- inviscid & Riemann fluxes ---------------------

def inviscidFluxes(U, g):
    F = cp.empty_like(U)
    u = U[1] / U[0]
    v = U[2] / U[0]
    p = (U[3] - 0.5 * U[0] * (u * u + v * v)) * (gamma - 1.0)
    u_convec = u * g[0] + v * g[1]
    F[0] = U[0] * u_convec
    F[1] = U[1] * u_convec + p * g[0]
    F[2] = U[2] * u_convec + p * g[1]
    F[3] = (U[3] + p) * u_convec
    return F


def riemannFluxes(UL, UR, S, g):
    FL = inviscidFluxes(UL, g)
    FR = inviscidFluxes(UR, g)
    return 0.5 * (FR + FL) - 0.5 * S * (UR - UL)

# ------------------------------ WENO‑5 JS -------------------------------

def weno5JSInterp(fm2, fm1, f00, fp1, fp2):
    q0 =  0.375 * fm2 - 1.250 * fm1 + 1.875 * f00
    q1 = -0.125 * fm1 + 0.750 * f00 + 0.375 * fp1
    q2 =  0.375 * f00 + 0.750 * fp1 - 0.125 * fp2

    d20 = fm2 - 2.0 * fm1 + f00
    d21 = fm1 - 2.0 * f00 + fp1
    d22 = f00 - 2.0 * fp1 + fp2

    d10 = fm2 - 4.0 * fm1 + 3.0 * f00
    d11 = fm1 - fp1
    d12 = fp2 - 4.0 * fp1 + 3.0 * f00

    IS0 = (13.0 / 12.0) * d20 * d20 + 0.25 * d10 * d10
    IS1 = (13.0 / 12.0) * d21 * d21 + 0.25 * d11 * d11
    IS2 = (13.0 / 12.0) * d22 * d22 + 0.25 * d12 * d12

    tau2 = (IS0 - IS2) * (IS0 - IS2)

    a0 = 0.0625 * (1.0 / (IS0 * IS0 + 1e‑6))
    a1 = 0.6250 * (1.0 / (IS1 * IS1 + 1e‑6))
    a2 = 0.3125 * (1.0 / (IS2 * IS2 + 1e‑6))

    return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)

# ---------------------- staggered finite‑difference ops ------------------

def ddxStag(f, inv_dx):
    return inv_dx * ((75.0/64.0) * (cp.roll(f, -1, axis=0) - cp.roll(f, +1, axis=0))
                     - (25.0/384.0) * (cp.roll(f, -2, axis=0) - cp.roll(f, +2, axis=0))
                     + (3.0/640.0) * (cp.roll(f, -3, axis=0) - cp.roll(f, +3, axis=0)))

def ddyStag(f, inv_dy):
    return inv_dy * ((75.0/64.0) * (cp.roll(f, -1, axis=1) - cp.roll(f, +1, axis=1))
                     - (25.0/384.0) * (cp.roll(f, -2, axis=1) - cp.roll(f, +2, axis=1))
                     + (3.0/640.0) * (cp.roll(f, -3, axis=1) - cp.roll(f, +3, axis=1)))

# ------------------------------ RHS evaluator ----------------------------

def calcRHSRiemann(cvars_ddt, cvars_now, dx: float, dy: float):
    """GPU version – direct port, *not* performance‑optimised."""
    U0, U1, U2, U3 = cvars_now

    # allocate fluxes on GPU
    F0 = cp.empty_like(U0)
    F1 = cp.empty_like(U1)
    F2 = cp.empty_like(U2)
    F3 = cp.empty_like(U3)

    # primitives
    rho, u, v, T, _ = conservativeToPrimitive(U0, U1, U2, U3)

    # ---------------------- x‑direction fluxes -------------------------
    for i in range(N):
        im = (i - 1) % N  # frequent reuse
        ip = (i + 1) % N
        for j in range(N):
            # Roe averages at face (i‑½, j)
            u_avg, v_avg, c_avg, h_avg = roeAvg(rho[im, j], rho[i, j],
                                                u[im, j], u[i, j],
                                                v[im, j], v[i, j],
                                                T[im, j], T[i, j])

            # Gather 6‑point stencil of conservative variables ↦ char vars
            ch0 = cp.empty((6,), dtype=cp.float64)
            ch1 = cp.empty((6,), dtype=cp.float64)
            ch2 = cp.empty((6,), dtype=cp.float64)
            ch3 = cp.empty((6,), dtype=cp.float64)

            for s, offset in enumerate([-3, -2, -1, 0, +1, +2]):
                idx = (i + offset) % N
                ch0[s], ch1[s], ch2[s], ch3[s] = conservativeToCharacteristic(
                    U0[idx, j], U1[idx, j], U2[idx, j], U3[idx, j],
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])

            # Left state
            UL = cp.empty((4,), dtype=cp.float64)
            UL[0], UL[1], UL[2], UL[3] = characteristicToConservative(
                weno5JSInterp(ch0[0], ch0[1], ch0[2], ch0[3], ch0[4]),
                weno5JSInterp(ch1[0], ch1[1], ch1[2], ch1[3], ch1[4]),
                weno5JSInterp(ch2[0], ch2[1], ch2[2], ch2[3], ch2[4]),
                weno5JSInterp(ch3[0], ch3[1], ch3[2], ch3[3], ch3[4]),
                c_avg, u_avg, v_avg, h_avg, [1.0, 0.0], [0.0, 1.0])

            # Right state
            UR = cp.empty((4,), dtype=cp.float64)
            UR[0], UR[1], UR[2], UR[3] = characteristicToConservative(
                weno5JSInterp(ch0[5], ch0[4], ch0[3], ch0[2], ch0[1]),
                weno5JSInterp(ch1[5], ch1[4], ch1[3], ch1[2], ch1[1]),
                weno5JSInterp(ch2[5], ch2[4], ch2[3], ch2[2], ch2[1]),
                weno5JSInterp(ch3[5], ch3[4], ch3[3], ch3[2], ch3[1]),
                c_avg, u_avg, v_avg, h_avg, [1.0, 0.0], [0.0, 1.0])

            # Wave speed estimate (Rusanov)
            uL = UL[1] / UL[0]
            vL = UL[2] / UL[0]
            eL = UL[3] / UL[0] - 0.5 * (uL * uL + vL * vL)
            uR = UR[1] / UR[0]
            vR = UR[2] / UR[0]
            eR = UR[3] / UR[0] - 0.5 * (uR * uR + vR * vR)
            cL = cp.sqrt(gamma * eL * (gamma - 1.0))
            cR = cp.sqrt(gamma * eR * (gamma - 1.0))
            S  = cp.maximum(cp.abs(uL) + cL, cp.abs(uR) + cR)

            F0[i, j], F1[i, j], F2[i, j], F3[i, j] = riemannFluxes(UL, UR, S, [1.0, 0.0])

    # update RHS with x‑derivatives
    cvars_ddt[0][:] = ddxStag(F0, -1.0 / dx)
    cvars_ddt[1][:] = ddxStag(F1, -1.0 / dx)
    cvars_ddt[2][:] = ddxStag(F2, -1.0 / dx)
    cvars_ddt[3][:] = ddxStag(F3, -1.0 / dx)

    # ---------------------- y‑direction fluxes -------------------------
    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N
        for i in range(N):
            u_avg, v_avg, c_avg, h_avg = roeAvg(rho[i, jm], rho[i, j],
                                                u[i, jm], u[i, j],
                                                v[i, jm], v[i, j],
                                                T[i, jm], T[i, j])

            ch0 = cp.empty((6,), dtype=cp.float64)
            ch1 = cp.empty((6,), dtype=cp.float64)
            ch2 = cp.empty((6,), dtype=cp.float64)
            ch3 = cp.empty((6,), dtype=cp.float64)

            for s, offset in enumerate([-3, -2, -1, 0, +1, +2]):
                idx = (j + offset) % N
                ch0[s], ch1[s], ch2[s], ch3[s] = conservativeToCharacteristic(
                    U0[i, idx], U1[i, idx], U2[i, idx], U3[i, idx],
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])

            UL = cp.empty((4,), dtype=cp.float64)
            UL[0], UL[1], UL[2], UL[3] = characteristicToConservative(
                weno5JSInterp(ch0[0], ch0[1], ch0[2], ch0[3], ch0[4]),
                weno5JSInterp(ch1[0], ch1[1], ch1[2], ch1[3], ch1[4]),
                weno5JSInterp(ch2[0], ch2[1], ch2[2], ch2[3], ch2[4]),
                weno5JSInterp(ch3[0], ch3[1], ch3[2], ch3[3], ch3[4]),
                c_avg, u_avg, v_avg, h_avg, [0.0, 1.0], [1.0, 0.0])

            UR = cp.empty((4,), dtype=cp.float64)
            UR[0], UR[1], UR[2], UR[3] = characteristicToConservative(
                weno5JSInterp(ch0[5], ch0[4], ch0[3], ch0[2], ch0[1]),
                weno5JSInterp(ch1[5], ch1[4], ch1[3], ch1[2], ch1[1]),
                weno5JSInterp(ch2[5], ch2[4], ch2[3], ch2[2], ch2[1]),
                weno5JSInterp(ch3[5], ch3[4], ch3[3], ch3[2], ch3[1]),
                c_avg, u_avg, v_avg, h_avg, [0.0, 1.0], [1.0, 0.0])

            uL = UL[1] / UL[0]
            vL = UL[2] / UL[0]
            eL = UL[3] / UL[0] - 0.5 * (uL * uL + vL * vL)
            uR = UR[1] / UR[0]
            vR = UR[2] / UR[0]
            eR = UR[3] / UR[0] - 0.5 * (uR * uR + vR * vR)
            cL = cp.sqrt(gamma * eL * (gamma - 1.0))
            cR = cp.sqrt(gamma * eR * (gamma - 1.0))
            S  = cp.maximum(cp.abs(vL) + cL, cp.abs(vR) + cR)

            F0[i, j], F1[i, j], F2[i, j], F3[i, j] = riemannFluxes(UL, UR, S, [0.0, 1.0])

    # add y‑derivatives
    cvars_ddt[0][:] += ddyStag(F0, -1.0 / dy)
    cvars_ddt[1][:] += ddyStag(F1, -1.0 / dy)
    cvars_ddt[2][:] += ddyStag(F2, -1.0 / dy)
    cvars_ddt[3][:] += ddyStag(F3, -1.0 / dy)

# --------------------------- SSPRK‑3 integrator --------------------------

def SSPRK3Riemann(c0: ConsVars, c1: ConsVars, c2: ConsVars, dx: float, dy: float, dt: float):
    # stage 1
    calcRHSRiemann(c1.get(), c0.get(), dx, dy)
    c1.mass[:] = c0.mass + dt * c1.mass
    c1.mmtx[:] = c0.mmtx + dt * c1.mmtx
    c1.mmty[:] = c0.mmty + dt * c1.mmty
    c1.enrg[:] = c0.enrg + dt * c1.enrg

    # stage 2
    calcRHSRiemann(c2.get(), c1.get(), dx, dy)
    c2.mass[:] = 0.75 * c0.mass + 0.25 * c1.mass + 0.25 * dt * c2.mass
    c2.mmtx[:] = 0.75 * c0.mmtx + 0.25 * c1.mmtx + 0.25 * dt * c2.mmtx
    c2.mmty[:] = 0.75 * c0.mmty + 0.25 * c1.mmty + 0.25 * dt * c2.mmty
    c2.enrg[:] = 0.75 * c0.enrg + 0.25 * c1.enrg + 0.25 * dt * c2.enrg

    # stage 3
    calcRHSRiemann(c1.get(), c2.get(), dx, dy)
    c0.mass[:] = (1.0/3.0) * c0.mass + (2.0/3.0) * c2.mass + (2.0/3.0) * dt * c1.mass
    c0.mmtx[:] = (1.0/3.0) * c0.mmtx + (2.0/3.0) * c2.mmtx + (2.0/3.0) * dt * c1.mmtx
    c0.mmty[:] = (1.0/3.0) * c0.mmty + (2.0/3.0) * c2.mmty + (2.0/3.0) * dt * c1.mmty
    c0.enrg[:] = (1.0/3.0) * c0.enrg + (2.0/3.0) * c2.enrg + (2.0/3.0) * dt * c1.enrg

# ------------------------- initial condition ----------------------------

def setInitialCondition(cvars: ConsVars, x: cp.ndarray, y: cp.ndarray):
    rho = cp.ones_like(x)
    p   = cp.ones_like(x)

    flag = (cp.abs(x) + cp.abs(y)) / L
    idx1 = flag < 0.25
    idx2 = ~idx1

    rho[idx1] = 0.125
    rho[idx2] = 1.0
    p  [idx1] = 0.140
    p  [idx2] = 1.0

    cvars.mass[:] = rho
    cvars.mmtx[:] = 0.0
    cvars.mmty[:] = 0.0
    cvars.enrg[:] = p / (gamma - 1.0)

# --------------------------- I/O utilities ------------------------------

def visualizeSolution(f: cp.ndarray, fname: str):
    plt.figure()
    plt.imshow(cp.asnumpy(f.T), origin="lower", extent=(-0.5*L, 0.5*L, -0.5*L, 0.5*L))
    plt.colorbar()
