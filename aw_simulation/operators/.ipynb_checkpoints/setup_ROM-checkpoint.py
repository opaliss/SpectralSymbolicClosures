
import numpy as np
from operators.FOM import D_matrix_inv_full, D_matrix_full, A_matrix_off, A_matrix_diag, B, A_matrix_col
from operators.ROM import theta_matrix, xi_matrix, Z_matrix, J_matrix
import scipy.sparse


class SimulationSetup:
    def __init__(self, Nx, Nx_total, Nv, epsilon, alpha_e1, alpha_e2, alpha_i, u_e1, u_e2, u_i, L, dt, T0, T, nu,
                 M, Nr, alpha_tol, u_tol, Ur_e1, Ur_e2, Ur_i, m_e1=1, m_e2=1, m_i=1836, q_e1=-1, q_e2=-1, q_i=1):
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
        self.alpha_e1 = [alpha_e1]
        self.alpha_e2 = [alpha_e2]
        self.alpha_i = [alpha_i]
        # velocity scaling
        self.u_e1 = [u_e1]
        self.u_e2 = [u_e2]
        self.u_i = [u_i]
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
        # dimensionality parameters
        self.M = M
        self.Nr = Nr
        self.NF = self.M * self.Nx_total
        self.NK = (self.Nv - self.M) * self.Nx_total

        # matrices
        # Fourier derivative matrix
        self.D_inv = D_matrix_inv_full(Nx=self.Nx, L=self.L)
        self.D = D_matrix_full(Nx=self.Nx, L=self.L).todense()

        # projection matrices
        self.Ur_e1 = Ur_e1
        self.Ur_e2 = Ur_e2
        self.Ur_i = Ur_i

        # Fourier transform matrix
        self.Z = Z_matrix(Nx_total=self.Nx_total, Nv=self.Nv, Nx=self.Nx, M=self.M)
        self.J = J_matrix(Nx_total=self.Nx_total, Nv=self.Nv, M=self.M)

        # fluid matrices
        # matrix of coefficients (advection)
        self.A_diag_F = A_matrix_diag(Nv=self.Nv, D=self.D)[:self.NF, :self.NF]
        self.A_off_F = A_matrix_off(Nx_total=self.Nx_total, Nv=self.Nv, D=self.D)[:self.NF, :self.NF]
        self.A_col_F = A_matrix_col(Nx_total=self.Nx_total, Nv=self.Nv, nu=self.nu)[:self.NF, :self.NF]
        # matrix of coefficient (acceleration)
        self.B_F = B(Nv=self.Nv, Nx_total=self.Nx_total)[:self.NF, :self.NF]

        # kinetic matrices
        # matrix of coefficients (advection)
        A_diag_K = A_matrix_diag(Nv=self.Nv, D=self.D)[self.NF:, self.NF:]
        A_off_K = A_matrix_off(Nx_total=self.Nx_total, Nv=self.Nv, D=self.D)[self.NF:, self.NF:]
        A_col_K = A_matrix_col(Nx_total=self.Nx_total, Nv=self.Nv, nu=self.nu)[self.NF:, self.NF:]
        B_K = B(Nv=self.Nv, Nx_total=self.Nx_total)[self.NF:, self.NF:]

        # reduced diagonal A matrix
        self.A_diag_K_e1_reduced = self.get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_e1)
        self.A_diag_K_e2_reduced = self.get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_e2)
        self.A_diag_K_i_reduced = self.get_kinetic_reduced_A_matrix(A=A_diag_K, Ur=self.Ur_i)

        # reduced off diagonal A matrix
        self.A_off_K_e1_reduced = self.get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_e1)
        self.A_off_K_e2_reduced = self.get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_e2)
        self.A_off_K_i_reduced = self.get_kinetic_reduced_A_matrix(A=A_off_K, Ur=self.Ur_i)

        # reduced collisional diagonal A matrix
        self.A_col_K_e1_reduced = self.get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_e1)
        self.A_col_K_e2_reduced = self.get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_e2)
        self.A_col_K_i_reduced = self.get_kinetic_reduced_A_matrix(A=A_col_K, Ur=self.Ur_i)

        # matrix of coefficient (acceleration)
        self.B_K_e1_reduced = self.get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_e1)
        self.B_K_e2_reduced = self.get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_e2)
        self.B_K_i_reduced = self.get_kinetic_reduced_B_matrix(B=B_K, Ur=self.Ur_i)

        # sparse coupling matrices
        G_F = - np.sqrt(self.M / 2) * xi_matrix(Nx_total=self.Nx_total, Nv=self.M) @ self.D @ theta_matrix(Nx_total=self.Nx_total, Nv=self.Nv - self.M).T
        upsilon_K = np.sqrt(2 * M) * theta_matrix(Nx_total=self.Nx_total, Nv=self.Nv - self.M)

        self.G_e1_F_reduced = self.get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_e1)
        self.G_e2_F_reduced = self.get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_e2)
        self.G_i_F_reduced = self.get_fluid_reduced_G_matrix(G=G_F, Ur=self.Ur_i)

        self.G_e1_K_reduced = self.get_kinetic_reduced_G_matrix(G=G_F.T, Ur=self.Ur_e1)
        self.G_e2_K_reduced = self.get_kinetic_reduced_G_matrix(G=G_F.T, Ur=self.Ur_e2)
        self.G_i_K_reduced = self.get_kinetic_reduced_G_matrix(G=G_F.T, Ur=self.Ur_i)

        self.Upsilon_e1_K_reduced = self.get_kinetic_reduced_G_matrix(G=upsilon_K, Ur=self.Ur_e1)
        self.Upsilon_e2_K_reduced = self.get_kinetic_reduced_G_matrix(G=upsilon_K, Ur=self.Ur_e2)
        self.Upsilon_i_K_reduced = self.get_kinetic_reduced_G_matrix(G=upsilon_K, Ur=self.Ur_i)

    def add_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1.append(alpha_e1_curr)

    def add_alpha_e2(self, alpha_e2_curr):
        self.alpha_e2.append(alpha_e2_curr)

    def add_alpha_i(self, alpha_i_curr):
        self.alpha_i.append(alpha_i_curr)

    def add_u_e1(self, u_e1_curr):
        self.u_e1.append(u_e1_curr)

    def add_u_e2(self, u_e2_curr):
        self.u_e2.append(u_e2_curr)

    def add_u_i(self, u_i_curr):
        self.u_i.append(u_i_curr)

    def replace_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1[-1] = alpha_e1_curr

    def replace_alpha_e2(self, alpha_e2_curr):
        self.alpha_e2[-1] = alpha_e2_curr

    def replace_alpha_i(self, alpha_i_curr):
        self.alpha_i[-1] = alpha_i_curr

    def replace_u_e1(self, u_e1_curr):
        self.u_e1[-1] = u_e1_curr

    def replace_u_e2(self, u_e2_curr):
        self.u_e2[-1] = u_e2_curr

    def replace_u_i(self, u_i_curr):
        self.u_i[-1] = u_i_curr

    def get_kinetic_reduced_A_matrix(self, A, Ur):
        return np.conjugate(Ur).T @ A @ Ur

    def get_kinetic_reduced_B_matrix(self, B, Ur):
        return np.conjugate(Ur).T @ B @ self.Z @ self.J @ scipy.sparse.kron(np.eye(self.Nx_total), Ur, format="csr")

    def get_fluid_reduced_G_matrix(self, G, Ur):
        return G @ Ur

    def get_kinetic_reduced_G_matrix(self, G, Ur):
        return np.conjugate(Ur).T @ G