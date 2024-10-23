import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
from sympy import banded, symbols
import numpy as np
import scipy
import pickle


def factorial_ratio(n, m):
    # return n! / m!
    if n >= m:
        return np.prod(range(m + 1, n + 1))
    else:
        return 1 / np.prod(range(n + 1, m + 1))


# loop over velocity resolutions
for Nv in np.arange(4, 22, 2):
    # hypercollisionality order ~ n^{2alpha -1}
    # alpha = 1 (Lenard Bernstein 1958) ~n
    # alpha = 2 (Camporeale 2006) ~n^3
    for alpha in np.arange(1, 6):

        # symbolic variables
        xi = symbols('xi')
        # must be real and not complex
        nu = symbols('nu', real=True)
        # must be an integer from definition
        k = symbols('k', integer=True)

        # advection matrix (off-diagonal)
        vec = sympy.zeros(Nv)
        for jj in range(1, Nv + 1):
            vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

        # advection matrix (main-diagonal)
        const_factor = sympy.factorial(4)
        vec2 = sympy.zeros(Nv)
        for nn in range(0, Nv + 1):
            # hyper collisions coefficient
            vec2[nn] = sympy.Rational(factorial_ratio(n=jj, m=jj - 2 * alpha + 1),
                                      1 / factorial_ratio(n=Nv - 2 * alpha, m=Nv - 1))

        # enforce k=1 for simplicity now
        k = 1

        # create an advection tri-diagonal matrix
        A = banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1]), 0: tuple(nu * vec2[0, :] / (sympy.I * sympy.sqrt(2) * k))})

        # identity matrix
        I = np.eye(Nv, dtype=int)

        # invert matrix
        M = sympy.Matrix(I * xi - k / np.abs(k) * A)

        # get final response function
        R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2) * k / np.abs(k)))

        asymptotics_0 = R_approx.series(xi, 0, 2)

        func = sympy.lambdify(nu, asymptotics_0.coeff(xi, 1) + sympy.I * sympy.sqrt(sympy.pi), modules='numpy')
        sol_coeff = scipy.optimize.newton(func, x0=1, maxiter=20000, rtol=1e-3, full_output=True)

        # save optimal nu (for k=1)
        with open("optimal_nu_hyper_" + str(alpha) + "/nu_" + str(Nv) + ".txt", "wb") as outf:
            pickle.dump(sol_coeff[0], outf)

            # save optimal R(nu*) (for k=1)
            with open("optimal_R_hyper_" + str(alpha) + "/R_" + str(Nv) + ".txt", "wb") as outf:
                pickle.dump(sympy.simplify(R_approx.subs(nu, sol_coeff[0].real)), outf)

        print(sol_coeff)
        print("completed hypercollisional operator")
        print("Nv = ", Nv)
        print("alpha = ", alpha)
