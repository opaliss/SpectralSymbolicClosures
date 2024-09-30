import numpy as np
from operators.FOM import D_matrix_inv_full, D_matrix_full, A_matrix_off, A_matrix_diag, B, A_matrix_col


class SimulationSetupFOM:
    def __init__(self, Nx, Nx_total, Nv, epsilon, alpha_e, alpha_i, u_e, u_i, L, dt, T0, T, nu, col_type, closure_type,
                 m_e=1, m_i=1836, q_e=-1, q_i=1, ions=False):
        # set up configuration parameters
        # number of mesh points in x
        self.Nx = Nx
        # total number of points in x
        self.Nx_total = Nx_total
        # number of spectral expansions
        self.Nv = Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling of electron and ion
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i
        # velocity scaling
        self.u_e = u_e
        self.u_i = u_i
        # x grid is from 0 to L
        self.L = L
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e = m_e
        self.m_i = m_i
        # charge normalized
        self.q_e = q_e
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu
        # type of collisional operator
        self.col_type = col_type
        self.closure_type = closure_type

        # matrices
        # Fourier derivative matrix
        self.D_inv = D_matrix_inv_full(Nx=self.Nx, L=self.L)
        self.D = D_matrix_full(Nx=self.Nx, L=self.L).todense()

        # matrix of coefficients (advection)
        A_diag = A_matrix_diag(Nv=self.Nv, D=self.D)
        A_off = A_matrix_off(M0=0, MF=self.Nv, D=self.D, closure_type=self.closure_type)
        A_col = A_matrix_col(Nx_total=self.Nx_total, M0=0, MF=self.Nv, Nv=self.Nv, col_type=self.col_type)

        self.A_e = self.alpha_e * A_off + self.u_e * A_diag + self.nu * A_col
        if ions:
            self.A_i = self.alpha_i * A_off + self.u_i * A_diag + self.nu * A_col

        # matrix of coefficient (acceleration)
        self.B_e = (self.q_e / (self.m_e * self.alpha_e)) * B(Nx_total=self.Nx_total, M0=0, MF=self.Nv)
        if ions:
            self.B_i = (self.q_i / (self.m_i * self.alpha_i)) * B(Nx_total=self.Nx_total, M0=0, MF=self.Nv)

