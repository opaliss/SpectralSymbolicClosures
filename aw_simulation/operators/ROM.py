"""Operators of the Spectral Plasma Solver (ROM) Hermite-Fourier Expansion

Author: Opal Issan (oissan@ucsd.edu)
Date: December 11th, 2023
"""
import numpy as np
import scipy


def theta_matrix(Nx_total, Nv):
    """construct sparse matrix K

    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: sparse matrix K
    """
    theta = np.zeros((Nv * Nx_total, Nx_total), dtype="complex128")
    theta[:Nx_total, :Nx_total] = np.identity(Nx_total)
    return scipy.sparse.csr_matrix(theta)


def xi_matrix(Nx_total, Nv):
    """construct sparse matrix K

    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: sparse matrix K
    """
    theta = np.zeros((Nv * Nx_total, Nx_total), dtype="complex128")
    theta[(Nv-1) * Nx_total:, :Nx_total] = np.identity(Nx_total)
    return scipy.sparse.csr_matrix(theta)


def K_matrix_index(Nx_total, Nv, ii, M):
    """construct sparse matrix K

    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :param: ii: int, index to locate identity
    :param M: int, number of fluid moments to solve unperturbed
    :return: sparse matrix K
    """
    K = np.zeros(((Nv - M) * Nx_total, Nx_total), dtype=int)
    K[Nx_total*ii: Nx_total*(ii+1), :Nx_total] = np.identity(n=Nx_total, dtype=int)
    return scipy.sparse.csr_matrix(K)


def Z_block(Nx_total, Nx):
    """

    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nx: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :param M: int, number of fluid moments to solve directly
    :return:
    """
    # Fourier transform matrix
    W = scipy.linalg.dft(n=2*Nx_total - 1, scale=None)
    # sparse matrix 1
    Phi_1 = np.zeros((2*Nx_total - 1, Nx_total))
    Phi_1[:Nx_total, :Nx_total] = np.identity(Nx_total)
    Phi_1 = scipy.sparse.dia_matrix(Phi_1)
    # sparse matrix 2
    Phi_2 = np.zeros((Nx_total, 2*Nx_total - 1))
    Phi_2[:Nx_total, Nx:Nx + Nx_total] = np.identity(Nx_total)
    Phi_2 = scipy.sparse.dia_matrix(Phi_2)
    # sparse matrix Z block
    return 1 / (2*Nx_total - 1) * Phi_2 @ np.conjugate(W).T @ scipy.linalg.khatri_rao(a=(W @ Phi_1).T, b=(W @ Phi_1).T).T


def J_matrix(Nx_total, Nv, M):
    """pre-ordering of Kronecker product

    :param Nx_total: int, number of Fourier spectral expansion coefficients (total is 2Nx + 1)
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :param M: int, number of fluid moments to solve unperturbed
    :return: E \otimes psi ordering
    """
    K = K_matrix_index(Nx_total=Nx_total, Nv=Nv, ii=0, M=M).T
    J = scipy.sparse.kron(scipy.sparse.identity(Nx_total, dtype=int), K, format="bsr")
    for jj in range(1, Nv-M):
        K = K_matrix_index(Nx_total=Nx_total, Nv=Nv, ii=jj, M=M).T
        J = scipy.sparse.vstack((J, scipy.sparse.kron(scipy.sparse.identity(Nx_total, dtype=int), K)),
                                format="bsr")
    return scipy.sparse.csr_matrix(J)


def total_mass(psi, alpha_e, alpha_i, Nv, Nx, L):
    """ N(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :return: N(t)
    """
    return L * (alpha_e * psi[Nx] + alpha_i * psi[Nv * (2 * Nx + 1) + Nx])


def total_mass_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L):
    """ N(t)

    :param psi: array, array of all coefficients of size (3*(Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 2
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :return: N(t)
    """
    return L * (alpha_e1 * psi[Nx]
                + alpha_e2 * psi[Nv * (2 * Nx + 1) + Nx]
                + alpha_i * psi[2 * Nv * (2 * Nx + 1) + Nx])


def total_momentum(psi, alpha_e, alpha_i, Nv, Nx, L, m_i, m_e, u_e, u_i):
    """P(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param L: float, length of spatial domain
    :param m_i: float, mass of ions (normalized to electron)
    :param m_e: float, mass of electrons  (normalized to electron, i.e. 1)
    :param u_e: float, velocity shifting parameter of electrons
    :param u_i: float, velocity shifting parameter of ions
    :param Nx: int, number of Fourier spectral terms
    :return: P(t)
    """
    electron_momentum = m_e * alpha_e * L * (alpha_e * psi[(2 * Nx + 1) + Nx] / np.sqrt(2) + u_e * psi[Nx])
    ion_momentum = m_i * alpha_i * L * (alpha_i * psi[Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + u_i * psi[Nv * (2 * Nx + 1) + Nx])
    return electron_momentum + ion_momentum


def total_momentum_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L, m_i, m_e1, m_e2, u_e1, u_e2, u_i):
    """P(t)

    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 2
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param L: float, length of spatial domain
    :param m_i: float, mass of ions (normalized to electron)
    :param m_e1: float, mass of electrons species 1 (normalized to electron, i.e. 1)
    :param m_e2: float, mass of electrons species 2 (normalized to electron, i.e. 1)
    :param u_e1: float, velocity shifting parameter of electrons species 1
    :param u_e2: float, velocity shifting parameter of electrons species 2
    :param u_i: float, velocity shifting parameter of ions
    :param Nx: int, number of Fourier spectral terms
    :return: P(t)
    """
    electron1_momentum = m_e1 * alpha_e1 * L * (alpha_e1 * psi[(2 * Nx + 1) + Nx] / np.sqrt(2) + u_e1 * psi[Nx])
    electron2_momentum = m_e2 * alpha_e2 * L * (alpha_e2 * psi[Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                                + u_e2 * psi[Nv * (2 * Nx + 1) + Nx])
    ion_momentum = m_i * alpha_i * L * (alpha_i * psi[2 * Nv * (2 * Nx + 1) + (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + u_i * psi[2 * Nv * (2 * Nx + 1) + Nx])
    return electron1_momentum + electron2_momentum + ion_momentum


def total_energy_k(psi, alpha_e, alpha_i, Nv, Nx, L, u_e, u_i, m_e, m_i):
    """E_{k}(t)

    :param m_i: float, mass of ions (normalized to electron)
    :param m_e: float, mass of electrons  (normalized to electron, i.e. 1)
    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e: float, velocity scaling of electrons
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :param u_e: float, velocity shifting parameter of electrons
    :param u_i: float, velocity shifting parameter of ions
    :return: E_{k}(t)
    """
    # electron kinetic energy
    electron_kin = 0.5 * L * alpha_e * (alpha_e ** 2 * psi[2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                        + np.sqrt(2) * u_e * alpha_e * psi[(2 * Nx + 1) + Nx]
                                        + ((alpha_e ** 2) / 2 + u_e ** 2) * psi[Nx])

    # ion kinetic energy
    ion_kin = 0.5 * L * alpha_i * (alpha_i ** 2 * psi[(2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                   + np.sqrt(2) * u_i * alpha_i * psi[(2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                   + ((alpha_i ** 2) / 2 + u_i ** 2) * psi[(2 * Nx + 1) * Nv + Nx])
    return m_e * electron_kin + m_i * ion_kin


def total_energy_k_two_stream(psi, alpha_e1, alpha_e2, alpha_i, Nv, Nx, L, u_e1, u_e2, u_i, m_e1, m_e2, m_i):
    """E_{k}(t)

    :param m_i: float, mass of ions (normalized to electron)
    :param m_e1: float, mass of electrons species 1 (normalized to electron, i.e. 1)
    :param m_e2: float, mass of electrons species 2 (normalized to electron, i.e. 1)
    :param psi: array, array of all coefficients of size ((Nv)*(2Nx + 1))
    :param alpha_e1: float, velocity scaling of electrons species 1
    :param alpha_e2: float, velocity scaling of electrons species 1
    :param alpha_i: float, velocity scaling of ions
    :param Nv: int, total number of Hermite spectral terms (Nv)
    :param Nx: int, number of Fourier spectral terms
    :param L: float, length of spatial domain
    :param u_e1: float, velocity shifting parameter of electrons species 1
    :param u_e2: float, velocity shifting parameter of electrons species 2
    :param u_i: float, velocity shifting parameter of ions
    :return: E_{k}(t)
    """
    # electron species 1 kinetic energy
    electron1_kin = 0.5 * L * alpha_e1 * (alpha_e1 ** 2 * psi[2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                          + np.sqrt(2) * u_e1 * alpha_e1 * psi[(2 * Nx + 1) + Nx]
                                          + ((alpha_e1 ** 2) / 2 + u_e1 ** 2) * psi[Nx])

    # electron species 2 kinetic energy
    electron2_kin = 0.5 * L * alpha_e2 * (alpha_e2 ** 2 * psi[(2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                          + np.sqrt(2) * u_e2 * alpha_e2 * psi[(2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                          + ((alpha_e2 ** 2) / 2 + u_e2 ** 2) * psi[(2 * Nx + 1) * Nv + Nx])

    # ion kinetic energy
    ion_kin = 0.5 * L * alpha_i * (alpha_i ** 2 * psi[2 * (2 * Nx + 1) * Nv + 2 * (2 * Nx + 1) + Nx] / np.sqrt(2)
                                   + np.sqrt(2) * u_i * alpha_i * psi[2 * (2 * Nx + 1) * Nv + (2 * Nx + 1) + Nx]
                                   + ((alpha_i ** 2) / 2 + u_i ** 2) * psi[2 * (2 * Nx + 1) * Nv + Nx])

    return m_e1 * electron1_kin + m_e2 * electron2_kin + m_i * ion_kin
