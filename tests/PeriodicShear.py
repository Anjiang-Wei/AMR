import matplotlib
matplotlib.use("agg")
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from numba import jit, float64, int16, int32, prange
from numba.experimental import jitclass

t_final = 1.5
CFL = 0.45
dt_output = 1e-3
fig_name_fmt = "./ShearFigs/vis_shear_{:06d}.pdf"
dat_name_fmt = "./ShearData/dat_shear_{:06d}.h5"

Mc    = 0.7
r_rho = 2.0
Rg    = 1.0
gamma = 1.4
L     = 0.6
N     = 1024
k_wav = int(5)


def main():
    cvars_0 = ConsVars(N)
    cvars_1 = ConsVars(N)
    cvars_2 = ConsVars(N)

    dx = float(L) / N
    dy = float(L) / N
    x1d = np.linspace(-0.5 * L, 0.5 * L, num=N, endpoint=False) + 0.5 * dx
    y1d = np.linspace(-0.5 * L, 0.5 * L, num=N, endpoint=False) + 0.5 * dy
    x, y = np.meshgrid(x1d, y1d, indexing='ij')

    setInitialCondition(cvars_0, x, y)
    fig_count = 0
    visualizeSolution(cvars_0.mass, fig_name_fmt.format(fig_count))
    saveSolution(cvars_0.convertToPrim(), ['rho', 'u', 'v', 'T', 'p'], dat_name_fmt.format(fig_count))
    t = 0.0
    while (t < (t_final - 1e-14)):
        t_output = dt_output * (1 + fig_count)
        dt = min(t_final - t, t_output - t)
        _, u, v, T, p = cvars_0.convertToPrim()
        c = np.sqrt(gamma * Rg * T)
        U = max(np.max(np.abs(u) + c), np.max(np.abs(v) + c))
        dt = min(dt, CFL * dx / U)
        SSPRK3Riemann(cvars_0, cvars_1, cvars_2, dx, dy, dt)
        if (np.abs(t_output - dt - t) < 1e-14):
            fig_count += 1
            #visualizeSolution(cvars_0.mass, fig_name_fmt.format(fig_count))
            visualizeSchlieren(cvars_0.mass, dx, dy, fig_name_fmt.format(fig_count))
            saveSolution(cvars_0.convertToPrim(), ['rho', 'u', 'v', 'T', 'p'], dat_name_fmt.format(fig_count))
        t += dt
        print("Completed {:.2f}%, dt = {:12.5e}, t = {:12.5e}.".format(100.0 * t / t_final, dt, t))
    

def visualizeSchlieren(rho, dx, dy, fig_name : str):
    N  = rho.shape[0]
    i  = np.arange(N)
    im = (i - 1 + N) % N
    ip = (i + 1    ) % N
    drdx = 0.5 * (rho[ip, :] - rho[im, :]) / dx
    drdy = 0.5 * (rho[:, ip] - rho[:, im]) / dy
    dr   = np.sqrt(drdx * drdx + drdy * drdy)
    dr[:]/= np.max(dr)
    plt.figure()
    plt.imshow(np.transpose(dr), origin='lower', cmap='gray_r', extent=(-0.5*L, 0.5*L, -0.5*L, 0.5*L), vmin=0, vmax=0.8)
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()


def visualizeSolution(f, fig_name : str):
    plt.figure()
    plt.imshow(np.transpose(f), origin='lower', extent=(-0.5*L, 0.5*L, -0.5*L, 0.5*L))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()

def saveSolution(fields, field_names, filename : str):
    with h5py.File(filename, 'w') as f:
        for i in range(len(fields)):
            f.create_dataset(field_names[i], data=fields[i], dtype='f8')
            print("Saved field {:s}: min = {:.3e}; max = {:.3e}".format(field_names[i], np.min(fields[i]), np.max(fields[i])))


class ConsVars:
    def __init__(self, N):
        self.N    = N
        self.mass = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.mmtx = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.mmty = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.enrg = np.empty((N, N), dtype=np.dtype(np.float64, align=True))

    def get(self):
        return [self.mass, self.mmtx, self.mmty, self.enrg]

    def convertToPrim(self):
        rho = np.copy(self.mass)
        u   = self.mmtx / self.mass
        v   = self.mmty / self.mass
        T   = (self.enrg / self.mass - 0.5 * (u * u + v * v)) * (gamma - 1.0) / Rg
        p   = rho * Rg * T
        return rho, u, v, T, p


@jit(nopython=True)
def conservativeToPrimitive(mass, mmtx, mmty, enrg, gamma, Rg):
        rho = np.copy(mass)
        u   = mmtx / mass
        v   = mmty / mass
        T   = (enrg / mass - 0.5 * (u * u + v * v)) * (gamma - 1.0) / Rg
        p   = rho * Rg * T
        return rho, u, v, T, p


def setInitialCondition(cvars : ConsVars, x, y):
    U    = 1.0
    P    = 1.0 / gamma
    rho2 = (0.5 * Mc / U * (np.sqrt(1./r_rho) + 1)) ** 2
    rho1 = r_rho * rho2

    rho = np.ones(x.shape, dtype=np.float64)
    idx1 = np.where(np.abs(y) <  0.25 * L)
    idx2 = np.where(np.abs(y) >= 0.25 * L)

    rho[idx1] = rho1
    rho[idx2] = rho2
    
    T = P / (Rg * rho)

    dy = y[0, 1] - y[0, 0]
    mask_H = 1e-4 * np.exp(-(0.5 * (y - 0.25*L) / dy)**2)
    mask_L = 1e-4 * np.exp(-(0.5 * (y + 0.25*L) / dy)**2)

    u = np.zeros_like(rho)
    u[idx1] =  U
    u[idx2] = -U
    v = mask_H * np.sin(2.0 * np.pi * k_wav * x / L) + mask_L * np.cos(2.0 * np.pi * k_wav * x / L)
    

    cvars.mass[:] =  rho
    cvars.mmtx[:] =  rho * u
    cvars.mmty[:] =  rho * v
    cvars.enrg[:] = rho * (Rg * T / (gamma - 1.0) + 0.5 * (u * u + v * v))


@jit(nopython=True)
def roeAvg(rhoL, rhoR, uL, uR, vL, vR, TL, TR):
    rhoL_sqrt = rhoL ** 0.5
    rhoR_sqrt = rhoR ** 0.5
    rho_fact  = 1.0 / (rhoL_sqrt + rhoR_sqrt)

    cv = Rg / (gamma - 1.0)
    qL = uL * uL + vL * vL
    qR = uR * uR + vR * vR
    eL = cv * TL + 0.5 * qL 
    eR = cv * TR + 0.5 * qR 
    hL = eL + Rg * TL 
    hR = eR + Rg * TR

    rho_avg   = rhoL_sqrt * rhoR_sqrt
    u_avg     = (rhoL_sqrt * uL + rhoR_sqrt * uR) * rho_fact
    v_avg     = (rhoL_sqrt * vL + rhoR_sqrt * vR) * rho_fact
    e_avg     = (rhoL_sqrt * eL + rhoR_sqrt * eR) * rho_fact
    h_avg     = (rhoL_sqrt * hL + rhoR_sqrt * hR) * rho_fact
    c_avg     = ((gamma - 1.0) * (h_avg - 0.5 * (u_avg * u_avg + v_avg * v_avg))) ** 0.5

    return u_avg, v_avg, c_avg, h_avg

@jit(nopython=True)
def inviscidFluxes(U, g):
    '''
    Assemble Euler fluxe
    Params:
        U - vector of conservative variables
        g - unit normal vectors
    '''
    F = np.empty_like(U)
    u = U[1] / U[0]
    v = U[2] / U[0]
    p = (U[3] - 0.5 * U[0] * (u * u + v * v)) * (gamma - 1.0)
    u_convec = u * g[0] + v * g[1]
    F[0] = U[0] * u_convec
    F[1] = U[1] * u_convec + p * g[0]
    F[2] = U[2] * u_convec + p * g[1]
    F[3] = (U[3] + p) * u_convec
    return F
    

@jit(nopython=True)
def riemannFluxes(UL, UR, S, g):
    FL = inviscidFluxes(UL, g)
    FR = inviscidFluxes(UR, g)
    F = 0.5 * (FR + FL) - 0.5 * S * (UR - UL)
    return F


@jit(nopython=True)
def characteristicToConservative(ch0, ch1, ch2, ch3, c_avg, u_avg, v_avg, h_avg, g, l):
    '''
    Convert characteristic variables to conservative variables
    Params:
    ch0, ch1, ch2, ch3 -- characteristic variables in 2D
    U0, U1, U2, U3     -- conservative variables in 2D
    c_avg              -- Roe-Pike-averaged speed of sound
    u_avg              -- Roe-Pike-averaged x-velocity
    v_avg              -- Roe-Pike-averaged y-velocity
    g                  -- 2D unit normal vector
    l                  -- 2D unit tangential vector
    '''
    Ug = u_avg * g[0] + v_avg * g[1]
    Ul = u_avg * l[0] + v_avg * l[1]
    ek_avg = 0.5 * (u_avg * u_avg + v_avg * v_avg)
    U0 = ch0 + ch2 + ch3
    U1 = ch0 * (u_avg - c_avg * g[0]) + ch1 * l[0] + ch2 * u_avg + ch3 * (u_avg + c_avg * g[0])
    U2 = ch0 * (v_avg - c_avg * g[1]) + ch1 * l[1] + ch2 * v_avg + ch3 * (v_avg + c_avg * g[1])
    U3 = ch0 * (h_avg - c_avg * Ug) + ch1 * Ul + ch2 * ek_avg + ch3 * (h_avg + c_avg * Ug)
    return U0, U1, U2, U3


@jit(nopython=True)
def conservativeToCharacteristic(U0, U1, U2, U3, c_avg, u_avg, v_avg, g, l):
    '''
    Convert conservative variables to characteristic variables
    Params:
    ch0, ch1, ch2, ch3 -- characteristic variables in 2D
    U0, U1, U2, U3     -- conservative variables in 2D
    c_avg              -- Roe-Pike-averaged speed of sound
    u_avg              -- Roe-Pike-averaged x-velocity
    v_avg              -- Roe-Pike-averaged y-velocity
    g                  -- 2D unit normal vector
    l                  -- 2D unit tangential vector
    '''
    ek_avg = 0.5 * (u_avg * u_avg + v_avg * v_avg)
    gm1byc2 = 0.5 * (gamma - 1.0) / (c_avg * c_avg)
    Ug = u_avg * g[0] + v_avg * g[1]
    Ul = u_avg * l[0] + v_avg * l[1]

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



@jit(nopython=True, parallel=True)
def calcRHSRiemann(cvars_ddt, cvars_now, dx : float, dy : float):
    '''
    Calculate RHS leaf task
    '''
    U0 = cvars_now[0]
    U1 = cvars_now[1]
    U2 = cvars_now[2]
    U3 = cvars_now[3]
    F0 = np.empty_like(U0)
    F1 = np.empty_like(U1)
    F2 = np.empty_like(U2)
    F3 = np.empty_like(U3)

    rho, u, v, T, p = conservativeToPrimitive(U0, U1, U2, U3, gamma, Rg)

    # Compute x-flux: The indices of edges align with the lower sides of the nodes
    for i in prange(N):
        for j in range(N):
            u_avg, v_avg, c_avg, h_avg = roeAvg(rho[i-1, j], rho[i, j], \
                                                u  [i-1, j], u  [i, j], \
                                                v  [i-1, j], v  [i, j], \
                                                T  [i-1, j], T  [i, j])
            ch0 = np.empty((6,), dtype=np.float64)
            ch1 = np.empty((6,), dtype=np.float64)
            ch2 = np.empty((6,), dtype=np.float64)
            ch3 = np.empty((6,), dtype=np.float64)

            im = (i - 3 + N) % N
            ch0[0], ch1[0], ch2[0], ch3[0] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])
            im = (i - 2 + N) % N
            ch0[1], ch1[1], ch2[1], ch3[1] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])
            im = (i - 1 + N) % N
            ch0[2], ch1[2], ch2[2], ch3[2] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])
            im = i
            ch0[3], ch1[3], ch2[3], ch3[3] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])
            im = (i + 1) % N
            ch0[4], ch1[4], ch2[4], ch3[4] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])
            im = (i + 2) % N
            ch0[5], ch1[5], ch2[5], ch3[5] = conservativeToCharacteristic( \
                    U0[im, j], U1[im, j], U2[im, j], U3[im, j], \
                    c_avg, u_avg, v_avg, [1.0, 0.0], [0.0, 1.0])


            UL = np.empty((4,), dtype=np.float64)
            UR = np.empty((4,), dtype=np.float64)

            ch0I = weno5JSInterp(ch0[0], ch0[1], ch0[2], ch0[3], ch0[4])
            ch1I = weno5JSInterp(ch1[0], ch1[1], ch1[2], ch1[3], ch1[4])
            ch2I = weno5JSInterp(ch2[0], ch2[1], ch2[2], ch2[3], ch2[4])
            ch3I = weno5JSInterp(ch3[0], ch3[1], ch3[2], ch3[3], ch3[4])
            UL[0], UL[1], UL[2], UL[3] = characteristicToConservative( \
                    ch0I, ch1I, ch2I, ch3I, c_avg, u_avg, v_avg, h_avg, [1.0, 0.0], [0.0, 1.0])

            ch0I = weno5JSInterp(ch0[5], ch0[4], ch0[3], ch0[2], ch0[1])
            ch1I = weno5JSInterp(ch1[5], ch1[4], ch1[3], ch1[2], ch1[1])
            ch2I = weno5JSInterp(ch2[5], ch2[4], ch2[3], ch2[2], ch2[1])
            ch3I = weno5JSInterp(ch3[5], ch3[4], ch3[3], ch3[2], ch3[1])
            UR[0], UR[1], UR[2], UR[3] = characteristicToConservative( \
                    ch0I, ch1I, ch2I, ch3I, c_avg, u_avg, v_avg, h_avg, [1.0, 0.0], [0.0, 1.0])

            uL = UL[1] / UL[0]
            vL = UL[2] / UL[0]
            eL = UL[3] / UL[0] - 0.5 * (uL * uL + vL * vL)
            uR = UR[1] / UR[0]
            vR = UR[2] / UR[0]
            eR = UR[3] / UR[0] - 0.5 * (uR * uR + vR * vR)
            cL = np.sqrt(gamma * eL * (gamma - 1.0))
            cR = np.sqrt(gamma * eR * (gamma - 1.0))
            S = max(abs(uL) + cL, abs(uR) + cR)
            F0[i,j], F1[i,j], F2[i,j], F3[i,j] = riemannFluxes(UL, UR, S, [1.0, 0.0])

    cvars_ddt[0][:] = ddxStag(F0, -1.0/dx)
    cvars_ddt[1][:] = ddxStag(F1, -1.0/dx)
    cvars_ddt[2][:] = ddxStag(F2, -1.0/dx)
    cvars_ddt[3][:] = ddxStag(F3, -1.0/dx)
            
    # Compute y-flux: The indices of edges align with the lower sides of the nodes
    for i in prange(N):
        for j in range(N):
            u_avg, v_avg, c_avg, h_avg = roeAvg(rho[i, j-1], rho[i, j], \
                                                u  [i, j-1], u  [i, j], \
                                                v  [i, j-1], v  [i, j], \
                                                T  [i, j-1], T  [i, j])
            ch0 = np.empty((6,), dtype=np.float64)
            ch1 = np.empty((6,), dtype=np.float64)
            ch2 = np.empty((6,), dtype=np.float64)
            ch3 = np.empty((6,), dtype=np.float64)

            jm = (j - 3 + N) % N
            ch0[0], ch1[0], ch2[0], ch3[0] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])
            jm = (j - 2 + N) % N
            ch0[1], ch1[1], ch2[1], ch3[1] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])
            jm = (j - 1 + N) % N
            ch0[2], ch1[2], ch2[2], ch3[2] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])
            jm = j
            ch0[3], ch1[3], ch2[3], ch3[3] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])
            jm = (j + 1) % N
            ch0[4], ch1[4], ch2[4], ch3[4] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])
            jm = (j + 2) % N
            ch0[5], ch1[5], ch2[5], ch3[5] = conservativeToCharacteristic( \
                    U0[i, jm], U1[i, jm], U2[i, jm], U3[i, jm], \
                    c_avg, u_avg, v_avg, [0.0, 1.0], [1.0, 0.0])


            UL = np.empty((4,), dtype=np.float64)
            UR = np.empty((4,), dtype=np.float64)

            ch0I = weno5JSInterp(ch0[0], ch0[1], ch0[2], ch0[3], ch0[4])
            ch1I = weno5JSInterp(ch1[0], ch1[1], ch1[2], ch1[3], ch1[4])
            ch2I = weno5JSInterp(ch2[0], ch2[1], ch2[2], ch2[3], ch2[4])
            ch3I = weno5JSInterp(ch3[0], ch3[1], ch3[2], ch3[3], ch3[4])
            UL[0], UL[1], UL[2], UL[3] = characteristicToConservative( \
                    ch0I, ch1I, ch2I, ch3I, c_avg, u_avg, v_avg, h_avg, [0.0, 1.0], [1.0, 0.0])

            ch0I = weno5JSInterp(ch0[5], ch0[4], ch0[3], ch0[2], ch0[1])
            ch1I = weno5JSInterp(ch1[5], ch1[4], ch1[3], ch1[2], ch1[1])
            ch2I = weno5JSInterp(ch2[5], ch2[4], ch2[3], ch2[2], ch2[1])
            ch3I = weno5JSInterp(ch3[5], ch3[4], ch3[3], ch3[2], ch3[1])
            UR[0], UR[1], UR[2], UR[3] = characteristicToConservative( \
                    ch0I, ch1I, ch2I, ch3I, c_avg, u_avg, v_avg, h_avg, [0.0, 1.0], [1.0, 0.0])

            uL = UL[1] / UL[0]
            vL = UL[2] / UL[0]
            eL = UL[3] / UL[0] - 0.5 * (uL * uL + vL * vL)
            uR = UR[1] / UR[0]
            vR = UR[2] / UR[0]
            eR = UR[3] / UR[0] - 0.5 * (uR * uR + vR * vR)
            cL = np.sqrt(gamma * eL * (gamma - 1.0))
            cR = np.sqrt(gamma * eR * (gamma - 1.0))
            S = max(abs(vL) + cL, abs(vR) + cR)
            F0[i,j], F1[i,j], F2[i,j], F3[i,j] = riemannFluxes(UL, UR, S, [0.0, 1.0])

    cvars_ddt[0][:] += ddyStag(F0, -1.0/dy)
    cvars_ddt[1][:] += ddyStag(F1, -1.0/dy)
    cvars_ddt[2][:] += ddyStag(F2, -1.0/dy)
    cvars_ddt[3][:] += ddyStag(F3, -1.0/dy)
            



def SSPRK3Riemann(cvars_0 : ConsVars, cvars_1 : ConsVars, cvars_2 : ConsVars, dx : float, dy : float, dt : float):
    calcRHSRiemann(cvars_1.get(), cvars_0.get(), dx, dy)
    cvars_1.mass[:] = cvars_0.mass + (dt * cvars_1.mass)
    cvars_1.mmtx[:] = cvars_0.mmtx + (dt * cvars_1.mmtx)
    cvars_1.mmty[:] = cvars_0.mmty + (dt * cvars_1.mmty)
    cvars_1.enrg[:] = cvars_0.enrg + (dt * cvars_1.enrg)

    calcRHSRiemann(cvars_2.get(), cvars_1.get(), dx, dy)
    cvars_2.mass[:] = (0.75 * cvars_0.mass) + (0.25 * cvars_1.mass) + (0.25 * dt * cvars_2.mass)
    cvars_2.mmtx[:] = (0.75 * cvars_0.mmtx) + (0.25 * cvars_1.mmtx) + (0.25 * dt * cvars_2.mmtx)
    cvars_2.mmty[:] = (0.75 * cvars_0.mmty) + (0.25 * cvars_1.mmty) + (0.25 * dt * cvars_2.mmty)
    cvars_2.enrg[:] = (0.75 * cvars_0.enrg) + (0.25 * cvars_1.enrg) + (0.25 * dt * cvars_2.enrg)

    calcRHSRiemann(cvars_1.get(), cvars_2.get(), dx, dy)
    cvars_0.mass[:] = ((1.0/3.0) * cvars_0.mass) + ((2.0/3.0) * cvars_2.mass) + ((2.0/3.0) * dt * cvars_1.mass)
    cvars_0.mmtx[:] = ((1.0/3.0) * cvars_0.mmtx) + ((2.0/3.0) * cvars_2.mmtx) + ((2.0/3.0) * dt * cvars_1.mmtx)
    cvars_0.mmty[:] = ((1.0/3.0) * cvars_0.mmty) + ((2.0/3.0) * cvars_2.mmty) + ((2.0/3.0) * dt * cvars_1.mmty)
    cvars_0.enrg[:] = ((1.0/3.0) * cvars_0.enrg) + ((2.0/3.0) * cvars_2.enrg) + ((2.0/3.0) * dt * cvars_1.enrg)



@jit(nopython=True)
def ddxStag(f, inv_dx : float):
    '''
    Calculate the derivative along x-dimension to backward position
    '''
    N = f.shape[0]
    im1 = np.arange(N)
    ip3 = (im1 + 3) % N
    ip2 = (im1 + 2) % N
    ip1 = (im1 + 1) % N
    im2 = (im1 - 1 + N) % N
    im3 = (im1 - 2 + N) % N
    #return inv_dx * ( (9.0/8.0) * (f[ip1, :] - f[im1, :]) - (1.0/24.0) * (f[ip2, :] - f[im2, :]))
    return inv_dx * ( (75.0/64.0) * (f[ip1, :] - f[im1, :]) - (25.0/384.0) * (f[ip2, :] - f[im2, :]) + (3.0/640.0) * (f[ip3, :] - f[im3, :]))


@jit(nopython=True)
def ddyStag(f, inv_dy : float):
    '''
    Calculate the derivative along y-dimension to backward position
    '''
    N = f.shape[1]
    jm1 = np.arange(N)
    jp3 = (jm1 + 3) % N
    jp2 = (jm1 + 2) % N
    jp1 = (jm1 + 1) % N
    jm2 = (jm1 - 1 + N) % N
    jm3 = (jm1 - 2 + N) % N
    #return inv_dy * ( (9.0/8.0) * (f[:, jp1] - f[:, jm1]) - (1.0/24.0) * (f[:, jp2] - f[:, jm2]))
    return inv_dy * ( (75.0/64.0) * (f[:, jp1] - f[:, jm1]) - (25.0/384.0) * (f[:, jp2] - f[:, jm2]) + (3.0/460.0) * (f[:, jp3] - f[:, jm3]))

@jit(nopython=True)
def weno5ZInterp(fm2, fm1, f00, fp1, fp2):
    '''
    5th-order WENO5-Z interpolation scheme.
    For more detail see Jiang & Shu, JCP (1996).
    Note: The coefficients used are based on interpolation not reconstruction.
    '''
    d0 = 0.0625  # linear weights of S0
    d1 = 0.6250  # linear weights of S1
    d2 = 0.3125  # linear weights of S2

    fs0 =  0.375 * fm2 - 1.250 * fm1 + 1.875 * f00
    fs1 = -0.125 * fm1 + 0.750 * f00 + 0.375 * fp1
    fs2 =  0.375 * f00 + 0.750 * fp1 - 0.125 * fp2

    # smoothness indicator: Eq(2.17)
    tmp1 = fm2 - 2.0 * fm1 +       f00
    tmp2 = fm2 - 4.0 * fm1 + 3.0 * f00
    beta0 = (13.0 / 12.0) * tmp1 * tmp1 + 0.25 * tmp2 * tmp2

    tmp1 = fm1 - 2.0 * f00 + fp1
    tmp2 = fm1             - fp1
    beta1 = (13.0 / 12.0) * tmp1 * tmp1 + 0.25 * tmp2 * tmp2

    tmp1 =       f00 - 2.0 * fp1 + fp2
    tmp2 = 3.0 * f00 - 4.0 * fp1 + fp2
    beta2 = (13.0 / 12.0) * tmp1 * tmp1 + 0.25 * tmp2 * tmp2

    tmp1 = (beta0 - beta2) * (beta0 - beta2)  # tau^2
    eps = 1.0e-6
    beta0 = d0 * (1.0 + tmp1 / (beta0 * beta0 + eps))
    beta1 = d1 * (1.0 + tmp1 / (beta1 * beta1 + eps))
    beta2 = d2 * (1.0 + tmp1 / (beta2 * beta2 + eps))

    return (beta0 * fs0 + beta1 * fs1 + beta2 * fs2) / (beta0 + beta1 + beta2)


@jit(nopython=True)
def weno5JSInterp(fm2, fm1, f00, fp1, fp2):
    '''
    5th-order WENO5-Z interpolation scheme.
    For more detail see Jiang & Shu, JCP (1996).
    Note: The coefficients used are based on interpolation not reconstruction.
    '''
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
    a0 = 0.0625 * (1.0 / (IS0 * IS0 + 1e-6))
    a1 = 0.6250 * (1.0 / (IS1 * IS1 + 1e-6))
    a2 = 0.3125 * (1.0 / (IS2 * IS2 + 1e-6))
    return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)

def create_output_dir():
    # ./ImplosionData/ and ./ImplosionFigs/
    os.makedirs("ImplosionData/", exist_ok=True)
    os.makedirs("ImplosionFigs/", exist_ok=True)

if __name__ == "__main__":
    create_output_dir()
    main()
