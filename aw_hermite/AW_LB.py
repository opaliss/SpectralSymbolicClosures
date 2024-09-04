import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import sympy
from sympy import banded, ones, Matrix, symbols, sqrt, print_latex, oo
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import wofz
import pickle



# number of moments
Nv = 10

# symbolic variables
xi = symbols('xi')
nu = symbols('nu', real = True) # must be real and not complex
k = symbols('k', integer = True) # must be an integer from defintion

# advection matrix (off-diagonal)
vec = sympy.zeros(Nv)
for jj in range(1, Nv+1):
    vec[jj-1] = sympy.sqrt(jj)/ (sympy.sqrt(2))


# advection matrix (main-diagonal)
vec2 = sympy.zeros(Nv)
for jj in range(0, Nv+1):
    # LB collisions coefficient
    vec2[jj] = jj


# enforce k=1 for simplicity now
k=1

# create an advection tri-diagonal matrix
A = banded({1: tuple(vec[0, :-1]), -1: tuple(vec[0, :-1]), 0: tuple(nu*vec2[0, :]/(sympy.I*sympy.sqrt(2)*k))})

# idenitity matrix
I = np.eye(Nv, dtype=int)

# invert matrix
M = sympy.Matrix(I*xi - k/ np.abs(k) * A)

# get final response function
R_approx = sympy.simplify(sympy.simplify(M.inv()[0, 1]/sympy.sqrt(2) * k / np.abs(k)))

# adiabatic limit expansion
asymptotics_0 = R_approx.series(xi, 0, 2)
func = sympy.lambdify(nu, asymptotics_0.coeff(xi, 1)+sympy.I*sympy.sqrt(sympy.pi), modules='numpy')
# best coefficient
sol_coeff = scipy.optimize.newton(func, x0=1, maxiter=20000, rtol=1e-3, full_output=True)

# save optimal nu (for k=1)
with open("optimal_nu_LB/nu_" + str(Nv) + ".txt", "wb") as outf:
    pickle.dump(sol_coeff[0], outf)

# save optimal R(nu*) (for k=1)
with open("optimal_R_LB/R_" + str(Nv) + ".txt", "wb") as outf:
    pickle.dump(sympy.simplify(R_approx.subs(nu, sol_coeff[0].real)), outf)

