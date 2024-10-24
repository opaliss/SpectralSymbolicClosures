import numpy as np
from operators.FOM import D_matrix_inv_full, D_matrix_full, A_matrix_off, A_matrix_diag, B, A_matrix_col


class SimulationSetupTwoStreamFOM:
    def __init__(self, Nx, Nx_total, Nv, epsilon, alpha_e1, alpha_e2,
                 alpha_i, u_e1, u_e2, u_i, L, dt, T0, T,
                 nu, n0_e1, n0_e2, col_type, hyper_rate, closure_type,
                 alpha_tol=np.inf, u_tol=np.inf, m_e1=1, m_e2=1, m_i=1836,
                 q_e1=-1, q_e2=-1, q_i=1, ions=True):
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
        self.alpha_e1 = alpha_e1
        self.alpha_e2 = alpha_e2
        self.alpha_i = alpha_i
        # velocity scaling
        self.u_e1 = u_e1
        self.u_e2 = u_e2
        self.u_i = u_i
        # average density coefficient
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
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
        self.m_e1 = m_e1
        self.m_e2 = m_e2
        self.m_i = m_i
        # charge normalized
        self.q_e1 = q_e1
        self.q_e2 = q_e2
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu
        # parameters tolerances
        self.u_tol = u_tol
        self.alpha_tol = alpha_tol

        # matrices
        # Fourier derivative matrix
        self.D_inv = D_matrix_inv_full(Nx=self.Nx, L=self.L)
        self.D = D_matrix_full(Nx=self.Nx, L=self.L).todense()

        # type of collisional operator
        self.col_type = col_type
        self.closure_type = closure_type
        self.hyper_rate = hyper_rate

        # matrix of coefficients (advection)
        A_diag = A_matrix_diag(Nv=self.Nv, D=self.D)
        A_off = A_matrix_off(M0=0, MF=self.Nv, D=self.D)
        A_col = A_matrix_col(Nx_total=self.Nx_total, M0=0, MF=self.Nv, Nv=self.Nv,
                             col_type=self.col_type, hyper_rate=self.hyper_rate)
        # A matrices
        self.A_e1 = self.u_e1 * A_diag + self.alpha_e1 * A_off + nu * A_col
        self.A_e2 = self.u_e2 * A_diag + self.alpha_e2 * A_off + nu * A_col

        # if ions evolve
        if ions:
            self.A_i = self.u_i * A_diag + self.alpha_i * A_off + nu * A_col

        # matrix of coefficient (acceleration)
        B_mat = B(M0=0, MF=self.Nv, Nx_total=self.Nx_total)

        # B matrices
        self.B_e1 = self.q_e1 / (self.m_e1 * self.alpha_e1) * B_mat
        self.B_e2 = self.q_e2 / (self.m_e2 * self.alpha_e2) * B_mat

        # if ions evolve
        if ions:
            self.B_i = self.q_i / (self.m_i * self.alpha_i) * B_mat

