"""
Operators of the Spectral Plasma Solver (FOM) Hermite-Fourier Expansion

Author: Opal Issan (oissan@ucsd.edu)
Date: March 4th, 2024
"""
import numpy as np
import scipy


def updated_alpha(alpha_prev, C20, C10, C00):
    """

    :param C00: float, the average zeroth moment
    :param C10: float, the average first moment
    :param C20: float, the average second moment
    :param alpha_prev: float, previous iterative alpha^{s}_{j-1} parameter
    :return: alpha at the updated iteration alpha^{s}_{j}
    """
    return (alpha_prev * np.sqrt(1 + np.sqrt(2) * C20 / C00 - (C10 / C00) ** 2)).real


def updated_u(u_prev, alpha_prev, C00, C10):
    """

    :param u_prev: float, previous iterative u^{s}_{j-1} parameter
    :param alpha_prev: float, previous iterative alpha^{s}_{j-1} parameter
    :param C00: float, the average zeroth moment
    :param C10: float, the average first moment
    :return: u at the updated iteration u^{s}_{j}
    """
    return (u_prev + alpha_prev * C10 / C00 / np.sqrt(2)).real


def a_constant(alpha_curr, alpha_prev):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j -1}
    :return: alpha^{s}_{j}/ alpha^{s}_{j-1}
    """
    return alpha_curr / alpha_prev


def b_constant(u_curr, u_prev, alpha_prev):
    """

    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j-1}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :return: (u^{s}_{j} - u^{s}_{j-1}) / alpha^{s}_{j-1}
    """
    return (u_curr - u_prev) / alpha_prev


def P_case_i(alpha_curr, alpha_prev, u_curr, u_prev, Nx_total, Nv):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j}
    :param Nx_total: int, total number of Fourier coefficients (2Nx+1 or Nx+1)
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # initialize P matrix
    P = np.zeros((Nv, Nv))

    # obtain a and b constants
    a = a_constant(alpha_curr=alpha_curr, alpha_prev=alpha_prev)
    a_square = a ** 2
    b = b_constant(u_curr=u_curr, u_prev=u_prev, alpha_prev=alpha_prev)

    """ Fill in the diagonal
    Inm = zeros(N); Jnm = zeros(N);
    Inm(1,1) = 1/a; Jnm(2,2) = 1/asq;
    M(1,1) = 1/a;
    """
    Inm, Jnm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))
    Inm[0, 0] = 1 / a
    Jnm[1, 1] = 1 / a_square
    P[0, 0] = 1 / a

    """
    for j=1:N-1
        M(j+1,j+1) = 1/a*M(j,j);
        n = j-1;
        Inm(n+2,1) = sqrt((n+1)/2)/(n+1)*(-2*b/a)*Inm(n+1,1);
        Jnm(n+3,2) = sqrt((n+2)/2)/(n+1)*(-2*b/a)*Jnm(n+2,2);
    end
    Jnm = Jnm(1:N,:);
    """
    for jj in range(1, Nv):
        P[jj, jj] = 1 / a * P[jj - 1, jj - 1]
        n = jj - 1
        Inm[n + 1, 0] = np.sqrt((n + 1) / 2) / (n + 1) * (-2 * b / a) * Inm[n, 0]
        if jj < Nv - 1:
            Jnm[n + 2, 1] = np.sqrt((n + 2) / 2) / (n + 1) * (-2 * b / a) * Jnm[n + 1, 1]

    """ Fill in the 1st column
    for n=2:N-1
        if mod(n,2) == 0; ev = n; else; ev = n-1; end
        for k=0:2:ev
            if k<N-2
                Inm(n+1,k+3) = (n-k-1)*(n-k)/(k/2+1)*(-2*b/a)^(-2)*(1/asq-1)*Inm(n+1,k+1);
            end
        end
    end
    M(:,1) = sum(Inm,2);
    """
    for n in range(2, Nv):
        if n % 2 == 0:
            ev = n
        else:
            ev = n - 1
        for k in np.arange(0, ev + 1, 2):
            if k < Nv - 2:
                Inm[n, k + 2] = (n - k - 1) * (n - k) / (k / 2 + 1) * (-2 * b / a) ** (-2) * (1 / a_square - 1) * Inm[
                    n, k]

    P[:, 0] = np.sum(Inm, axis=1)

    """ Fill in the odd columns
    for m=0:2:N-3
        Inmp = zeros(N);
        for k=m+2:2:N-1
            Inmp(m+2+2:end,k+1) = 2/asq/sqrt((m+2)*(m+1))*(k-m)/2*(1/asq-1)^(-1)*Inm(m+2+2:end,k+1);
        end
        Inm = Inmp;
        II =  sum(Inm,2);
        M(m+2+2:end,m+2+1) = II(m+2+2:end);
    end"""
    for m in range(0, Nv - 2, 2):
        Inmp = np.zeros((Nv, Nv))
        for k in np.arange(m + 2, Nv, 2):
            Inmp[m + 3:, k] = 2 / a_square / np.sqrt((m + 2) * (m + 1)) * (k - m) / 2 * (1 / a_square - 1) ** (
                -1) * Inm[m + 3:, k]
        Inm = Inmp
        II = np.sum(Inm, axis=1)
        P[m + 3:, m + 2] = II[m + 3:]

    """ Fill in the 2nd column
    for n=3:N-1
        if mod(n,2) == 0; ev = n-1; else; ev = n; end
        for k=1:2:ev
            if k<N-2
                Jnm(n+1,k+3) = (n-k-1)*(n-k)/((k-1)/2+1)*(-2*b/a)^(-2)*(1/asq-1)*Jnm(n+1,k+1);
            end
        end
    end
    M(:,2) = sum(Jnm,2);
    """
    for n in range(3, Nv):
        if n % 2 == 0:
            ev = n - 1
        else:
            ev = n
        for k in np.arange(1, ev + 1, 2):
            if k < Nv - 2:
                Jnm[n, k + 2] = (n - k - 1) * (n - k) / ((k - 1) / 2 + 1) * (-2 * b / a) ** (-2) * (
                        1 / a_square - 1) * Jnm[n, k]

    P[:, 1] = np.sum(Jnm, axis=1)
    """ Fill in the even columns
    for m=1:2:N-3
        Jnmp = zeros(N);
        for k=m+2:2:N-1
            Jnmp(m+2+2:end,k+1) = 2/asq/sqrt((m+2)*(m+1))*(k-m)/2*(1/asq-1)^(-1)*Jnm(m+2+2:end,k+1);
        end
        Jnm = Jnmp;
        JJ =  sum(Jnm,2);
        M(m+2+2:end,m+2+1) = JJ(m+2+2:end);
    end
    """
    for m in np.arange(1, Nv - 2, 2):
        Jnmp = np.zeros((Nv, Nv))
        for k in np.arange(m + 2, Nv, 2):
            Jnmp[m + 3:, k] = 2 / a_square / np.sqrt((m + 2) * (m + 1)) * (k - m) / 2 * (1 / a_square - 1) ** (-1) \
                              * Jnm[m + 3:, k]
        Jnm = Jnmp
        JJ = np.sum(Jnm, axis=1)
        P[m + 3:, m + 2] = JJ[m + 3:]
    return scipy.sparse.kron(np.tril(P), np.eye(N=Nx_total), format="bsr")


def P_case_ii(alpha_prev, u_curr, u_prev, Nx_total, Nv):
    """

    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j}
    :param Nx_total: int, total number of Fourier coefficients (2Nx+1 or Nx+1)
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # obtain b constant
    b = b_constant(u_curr=u_curr, u_prev=u_prev, alpha_prev=alpha_prev)

    """
    M = eye(N);
    for n=0:N-2 % fill in the 1st column
        M(n+2,1) = -sqrt(2)*b*sqrt(n+1)/(n+1)*M(n+1,1);
    end
    for m=2:N-1
        M(m+1:N,m) = -1/sqrt(2)/b/sqrt(m-1).*((m:N-1)-m+2).'.*M((m+1:N),m-1);
    end
    M = tril(M);
    """
    # initialize P
    P = np.eye(Nv)

    # loop over lower diagonal constants
    for n in range(0, Nv - 1):
        # fill in the 1st column
        P[n + 1, 0] = - np.sqrt(2) * b * np.sqrt(n + 1) / (n + 1) * P[n, 0]

    for m in range(2, Nv):
        P[m:, m - 1] = -1 / np.sqrt(2) / b / np.sqrt(m - 1) * (np.arange(m, Nv) - m + 2) * P[m:, m - 2]

    return scipy.sparse.kron(np.tril(P), np.eye(N=Nx_total), format="bsr")


def P_case_iii(alpha_curr, alpha_prev, Nx_total, Nv):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param Nx_total: int, total number of Fourier coefficients (2Nx+1 or Nx+1)
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # obtain a constant
    a = a_constant(alpha_curr=alpha_curr, alpha_prev=alpha_prev)
    a_squared = a ** 2
    """
    M = diag(1./a.^(1:N));
    for n=0:2:N-3 % fill in the first column
        M(n+3,1) = sqrt((n+1)/(n+2))*(1/asq-1)*M(n+1,1);
    end
    for m=1:N-3
        for n=3:N-1
            M(n+1,m+1) = m^(-1/2)*n^(1/2)/a*M(n,m);
        end
    end
    end
    """
    P = np.diag(1. / (a ** np.arange(1, Nv + 1)))
    for n in np.arange(0, Nv - 2, 2):
        # fill in the first column
        P[n + 2, 0] = np.sqrt((n + 1) / (n + 2)) * (1 / a_squared - 1) * P[n, 0]

    for m in range(1, Nv - 2):
        for n in range(3, Nv):
            P[n, m] = np.sqrt(n) / np.sqrt(m) / a * P[n - 1, m - 1]
    return scipy.sparse.kron(np.tril(P), np.eye(N=Nx_total), format="bsr")


def check_if_update_needed(u_s_curr, u_s, u_s_tol, alpha_s_curr, alpha_s, alpha_s_tol):
    """check if adaptive tolerance conditions are met

    :param u_s_curr: float, u^{s}_{j+1}
    :param u_s: float, u^{s}_{j}
    :param u_s_tol: float, u^{s}_{tol}
    :param alpha_s_curr: float, alpha^{s}_{j+1}
    :param alpha_s: float, alpha^{s}_{j}
    :param alpha_s_tol: float, alpha^{s}_{tol}
    :return: boolean (True/False)
    """
    if np.abs(u_s - u_s_curr) >= u_s_tol:
        return True
    elif np.abs(alpha_s - alpha_s_curr) / alpha_s >= alpha_s_tol:
        return True
    else:
        return False


def get_projection_matrix(u_s_curr, u_s, alpha_s_curr, alpha_s, Nx_total, Nv, alpha_s_tol, u_s_tol):
    """

    :param u_s_curr: float,  u^{s}_{j+1}
    :param u_s: float, U^{s}_{j}
    :param alpha_s_curr: float, alpha^{s}_{j+1}
    :param alpha_s: float, alpha^{s}_{j}
    :param Nx_total: int, total number of Fourier coefficients (2Nx+1 or Nx+1)
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # case (i)
    if np.abs(u_s_curr - u_s) >= u_s_tol and np.abs(alpha_s_curr - alpha_s) / alpha_s >= alpha_s_tol:
        return P_case_i(alpha_curr=alpha_s_curr, alpha_prev=alpha_s,
                        u_curr=u_s_curr, u_prev=u_s, Nx_total=Nx_total, Nv=Nv), 1

    # case (ii)
    elif np.abs(u_s_curr - u_s) > u_s_tol and np.abs(alpha_s_curr - alpha_s)/alpha_s < alpha_s_tol:
        return P_case_ii(alpha_prev=alpha_s, u_curr=u_s_curr, u_prev=u_s, Nx_total=Nx_total, Nv=Nv), 2

    # case (iii)
    elif np.abs(u_s_curr - u_s) < u_s_tol and np.abs(alpha_s_curr - alpha_s)/alpha_s > alpha_s_tol:
        return P_case_iii(alpha_curr=alpha_s_curr, alpha_prev=alpha_s, Nx_total=Nx_total, Nv=Nv), 3

    # no tolerance is met
    else:
        return np.eye(Nv*Nx_total)