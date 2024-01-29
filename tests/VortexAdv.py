####
## UNCOMMENT THIS WHILE RUNNING ON LINUX BASED CLUSTER
# import matplotlib
# matplotlib.use("agg")
####
import numpy as np
import matplotlib.pyplot as plt
import h5py


dt = 0.01
nsteps = 2000
fig_output_stride = 100
fig_name_fmt = "./VortexFigs/vis_{:06d}.pdf"

Rg    = 1.0
gamma = 1.4
L     = 12.0
N     = 64

def main():
    cvars_0 = ConsVars(N)
    cvars_1 = ConsVars(N)
    cvars_2 = ConsVars(N)

    x1d = np.linspace(-0.5 * L, 0.5 * L, num=N, endpoint=False)
    y1d = np.linspace(-0.5 * L, 0.5 * L, num=N, endpoint=False)
    x, y = np.meshgrid(x1d, y1d, indexing='ij')
    dx = float(L) / N
    dy = float(L) / N

    setInitialCondition(cvars_0, x, y)
    fig_count = 0
    visualizeSolution(cvars_0.mass, fig_name_fmt.format(fig_count))
    for tid in range(nsteps):
        SSPRK3(cvars_0, cvars_1, cvars_2, dx, dy, dt)
        if (((tid + 1) % fig_output_stride) == 0):
            fig_count += 1
            visualizeSolution(cvars_0.mass, fig_name_fmt.format(fig_count))
        print("Completed {:4d}/{:4d}, {:.2f}%.".format(tid+1, nsteps, 100.0*(tid+1)/nsteps))
    


def visualizeSolution(f, fig_name : str):
    plt.figure()
    plt.imshow(np.transpose(f), origin='lower', extent=(-0.5*L, 0.5*L, -0.5*L, 0.5*L))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()


class ConsVars:
    def __init__(self, N):
        self.N    = N
        self.mass = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.mmtx = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.mmty = np.empty((N, N), dtype=np.dtype(np.float64, align=True))
        self.enrg = np.empty((N, N), dtype=np.dtype(np.float64, align=True))


def setInitialCondition(cvars : ConsVars, x, y):
    alp  = 1.20
    eps  = 0.30
    U0   = 0.50
    V0   = 0.00
    r    = np.sqrt(x * x + y * y)
    G    = np.exp(alp * (1.0 - r * r))
    T0   = 1.0 / (gamma * Rg)
    AT   = (gamma - 1.0) * eps * eps / (4.0 * alp * gamma)
    isen = 1.0 / (gamma - 1.0)

    u   = U0 + r * eps * G * ( y / (r + ((r * r) < 1e-30)))
    v   = V0 + r * eps * G * (-x / (r + ((r * r) < 1e-30)))
    T   = T0 - AT * G * G
    rho = (1.0 - AT*G*G/T0) ** isen

    cvars.mass[:] = rho
    cvars.mmtx[:] = rho * u
    cvars.mmty[:] = rho * v
    cvars.enrg[:] = rho * (Rg * T / (gamma - 1.0) + 0.5 * (u * u + v * v))



def SSPRK3(cvars_0 : ConsVars, cvars_1 : ConsVars, cvars_2 : ConsVars, dx : float, dy : float, dt : float):
    calcRHS(cvars_1, cvars_0, dx, dy)
    cvars_1.mass[:] = cvars_0.mass + (dt * cvars_1.mass)
    cvars_1.mmtx[:] = cvars_0.mmtx + (dt * cvars_1.mmtx)
    cvars_1.mmty[:] = cvars_0.mmty + (dt * cvars_1.mmty)
    cvars_1.enrg[:] = cvars_0.enrg + (dt * cvars_1.enrg)

    calcRHS(cvars_2, cvars_1, dx, dy)
    cvars_2.mass[:] = (0.75 * cvars_0.mass) + (0.25 * cvars_1.mass) + (0.25 * dt * cvars_1.mass)
    cvars_2.mmtx[:] = (0.75 * cvars_0.mmtx) + (0.25 * cvars_1.mmtx) + (0.25 * dt * cvars_1.mmtx)
    cvars_2.mmty[:] = (0.75 * cvars_0.mmty) + (0.25 * cvars_1.mmty) + (0.25 * dt * cvars_1.mmty)
    cvars_2.enrg[:] = (0.75 * cvars_0.enrg) + (0.25 * cvars_1.enrg) + (0.25 * dt * cvars_1.enrg)

    calcRHS(cvars_1, cvars_2, dx, dy)
    cvars_0.mass[:] = ((1.0/3.0) * cvars_0.mass) + ((2.0/3.0) * cvars_2.mass) + ((2.0/3.0) * dt * cvars_1.mass)
    cvars_0.mmtx[:] = ((1.0/3.0) * cvars_0.mmtx) + ((2.0/3.0) * cvars_2.mmtx) + ((2.0/3.0) * dt * cvars_1.mmtx)
    cvars_0.mmty[:] = ((1.0/3.0) * cvars_0.mmty) + ((2.0/3.0) * cvars_2.mmty) + ((2.0/3.0) * dt * cvars_1.mmty)
    cvars_0.enrg[:] = ((1.0/3.0) * cvars_0.enrg) + ((2.0/3.0) * cvars_2.enrg) + ((2.0/3.0) * dt * cvars_1.enrg)


def calcRHS(cvars_ddt : ConsVars, cvars_now : ConsVars, dx : float, dy : float):
    '''
    Calculate RHS leaf task
    '''
    # Conservative to primitive
    rho_coll = cvars_now.mass
    u_coll   = cvars_now.mmtx / cvars_now.mass
    v_coll   = cvars_now.mmty / cvars_now.mass
    T_coll   = (cvars_now.enrg / cvars_now.mass - 0.5 * (u_coll * u_coll + v_coll * v_coll)) * (gamma - 1.0) / Rg
    p_coll   = rho_coll * Rg * T_coll

    # ASSEMBLE X - FLUXES
    u   = interpX(u_coll)
    v   = interpX(v_coll)
    T   = interpX(T_coll)
    p   = interpX(p_coll)
    rho = p / (Rg * T)
    H   = rho * (Rg * T * gamma / (gamma - 1.0) + 0.5 * (u * u + v * v)) 

    flux_mass = -rho     * u
    flux_mmtx = -rho * u * u - p
    flux_mmty = -rho * v * u
    flux_enrg = -H       * u

    cvars_ddt.mass[:] = ddxStag(flux_mass, 1.0 / dx)
    cvars_ddt.mmtx[:] = ddxStag(flux_mmtx, 1.0 / dx)
    cvars_ddt.mmty[:] = ddxStag(flux_mmty, 1.0 / dx)
    cvars_ddt.enrg[:] = ddxStag(flux_enrg, 1.0 / dx)

    # ASSEMBLE Y - FLUXES
    u   = interpY(u_coll)
    v   = interpY(v_coll)
    T   = interpY(T_coll)
    p   = interpY(p_coll)
    rho = p / (Rg * T)
    H   = rho * (Rg * T * gamma / (gamma - 1.0) + 0.5 * (u * u + v * v)) 

    flux_mass = -rho     * v
    flux_mmtx = -rho * u * v
    flux_mmty = -rho * v * v - p
    flux_enrg = -H       * v

    cvars_ddt.mass[:] += ddyStag(flux_mass, 1.0 / dy)
    cvars_ddt.mmtx[:] += ddyStag(flux_mmtx, 1.0 / dy)
    cvars_ddt.mmty[:] += ddyStag(flux_mmty, 1.0 / dy)
    cvars_ddt.enrg[:] += ddyStag(flux_enrg, 1.0 / dy)



def ddxStag(f, inv_dx : float):
    '''
    Calculate the derivative along x-dimension to backward position
    '''
    N = f.shape[0]
    ip1 = np.arange(N)
    ip2 = (ip1 + 1) % N
    im1 = (ip1 - 1 + N) % N
    im2 = (ip1 - 2 + N) % N
    return inv_dx * ( (9.0/8.0) * (f[ip1, :] - f[im1, :]) - (1.0/24.0) * (f[ip2, :] - f[im2, :]))


def ddyStag(f, inv_dy : float):
    '''
    Calculate the derivative along y-dimension to backward position
    '''
    N = f.shape[1]
    jp1 = np.arange(N)
    jp2 = (jp1 + 1) % N
    jm1 = (jp1 - 1 + N) % N
    jm2 = (jp1 - 2 + N) % N
    return inv_dy * ( (9.0/8.0) * (f[:, jp1] - f[:, jm1]) - (1.0/24.0) * (f[:, jp2] - f[:, jm2]))


def interpX(f):
    '''
    Interpolate along x-dimension to the forward position
    '''
    N = f.shape[0]
    im1 = np.arange(N)
    im2 = (im1 - 1 + N) % N 
    ip1 = (im1 + 1) % N
    ip2 = (im1 + 2) % N
    return (9.0 / 16.0) * (f[ip1, :] + f[im1, :]) - (1.0 / 16.0) * (f[ip2, :] + f[im2, :])



def interpY(f):
    '''
    Interpolate along y-dimension to the forward position
    '''
    N = f.shape[1]
    jm1 = np.arange(N)
    jm2 = (jm1 - 1 + N) % N 
    jp1 = (jm1 + 1) % N
    jp2 = (jm1 + 2) % N
    return (9.0 / 16.0) * (f[:, jp1] + f[:, jm1]) - (1.0 / 16.0) * (f[:, jp2] + f[:, jm2])




if __name__ == "__main__":
    main()
