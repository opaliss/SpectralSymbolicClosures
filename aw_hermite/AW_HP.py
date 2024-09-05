import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
from sympy import banded, ones, Matrix, symbols, sqrt, print_latex, oo
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import scipy
import pickle

# setup the number of Hermite moments
Nv = 16

# initialize the symbolic variables
xi = symbols('xi')
k = symbols('k', integer=True)
c = symbols('c', complex=True)
b = symbols('b', complex=True)
a = symbols('a', complex=True)

# create advection matrix A
vec = sympy.zeros(Nv)
for jj in range(1, Nv + 1):
    vec[jj - 1] = sympy.sqrt(jj) / (sympy.sqrt(2))

A = banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1])})
A[-1, Nv - 3] += sympy.I * a * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)
A[-1, Nv - 2] += sympy.I * b * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)
A[-1, Nv - 1] += sympy.I * c * sympy.sqrt(Nv) / sympy.sqrt(2) * k / np.abs(k)

# identity matrix
I = np.eye(Nv, dtype=int)

# eigenvalue matrix
M = sympy.Matrix(I * xi - k / np.abs(k) * A)

# inversion
R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1] / sympy.sqrt(2) * k / np.abs(k)))


# adiabatic limit matching
asymptotics_0 = R_approx.series(xi, 0, 4)
sol_coeff = sympy.solve([asymptotics_0.coeff(xi, 0) + 1,
                         asymptotics_0.coeff(xi, 1) + sympy.I*sympy.sqrt(sympy.pi),
                         asymptotics_0.coeff(xi, 2) -2], [a, b, c])


# save optimal (a, b, c)
with open("optimal_c_HP/coeff_" + str(Nv) + ".txt", "wb") as outf:
    pickle.dump(sol_coeff[0], outf)


# save optimal R(a, b, c)
with open("optimal_R_HP/R_" + str(Nv) + ".txt", "wb") as outf:
    pickle.dump(sympy.simplify(R_approx.subs([(a, sol_coeff[0][0]), (b, sol_coeff[0][1]), (c, sol_coeff[0][2])])), outf)