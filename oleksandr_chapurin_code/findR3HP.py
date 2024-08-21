from sympy import *
import math
init_printing()

n, u, T, q, S, x, d1, phi, m, n0, e, vt, sq_pi, X = symbols('n u T q S \zeta d1 phi m n0 e Vt sqrt(pi) X') # x = w/k, sq2 = sqrt(2)
T0 = m*vt**2
q = - 1j*sqrt(2)*X*n0*T
solution = solve([Eq(n*x - n0*u, 0),
	Eq(m*n0*x*u - (n0*T + T0*n + e*n0*phi), 0),
	Eq(x*n0*T - (2*n0*T0*u + q)/sq2/vt, 0)], [n, u, T])
#print solution
resp = -solution[n]*T0/e/phi/n0
print('R_3(HP) = ', simplify(resp))

#zsmall = 1j*sq_pi*exp(-x**2) - 2*x*(1 - 2*x**2/3)# + 4*x**4/15)
# resp_sers = resp.subs({z:zsmall})
resp_sers = resp.series(x,oo,5).removeO()
#resp_simp = simplify(resp_sers)
print('R_3(HP) = ', latex(resp_sers))
#print simplify(resp.series(x,0,2))
