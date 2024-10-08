"""Module to run the linear Landau damping testcase

Author: Opal Issan
Date: Oct 1st, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM import SimulationSetupFOM
import time
import numpy as np


def rhs(y):
    # electric field computed
    E = setup.D_inv @ charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i,
                                     q_e=setup.q_e, q_i=setup.q_i,
                                     C0_electron=y[:setup.Nx_total],
                                     C0_ions=C0_ions)

    # evolving only electrons
    return setup.A_e @ y + setup.B_e @  nonlinear_full(E=E, psi=y, Nv=setup.Nv, Nx_total=setup.Nx_total)


if __name__ == "__main__":
    setup = SimulationSetupFOM(Nx=10,
                               Nx_total=21,
                               Nv=20,
                               epsilon=1e-2,
                               alpha_e=np.sqrt(2),
                               alpha_i=np.sqrt(2 / 1836),
                               u_e=0,
                               u_i=0,
                               L=4 * np.pi,
                               dt=1e-2,
                               T0=0,
                               T=40,
                               nu=0,
                               col_type="collisionless",
                               closure_type="truncation")

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.Nv * setup.Nx_total, dtype="complex128")
    # electron equilibrium
    y0[setup.Nx] = 1 / setup.alpha_e
    # first electron perturbation
    y0[setup.Nx + 1] = 0.5 * setup.epsilon / setup.alpha_e
    y0[setup.Nx - 1] = 0.5 * setup.epsilon / setup.alpha_e
    # second electron perturbation
    y0[setup.Nx + 3] = 0.5 * setup.epsilon / setup.alpha_e
    y0[setup.Nx - 3] = 0.5 * setup.epsilon / setup.alpha_e


    # ions (unperturbed)
    C0_ions = np.zeros(setup.Nx_total)
    C0_ions[setup.Nx] = 1 / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                         right_hand_side=rhs,
                                                         r_tol=1e-8,
                                                         a_tol=1e-12,
                                                         max_iter=100,
                                                         param=setup,
                                                         adaptive=False)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # make directory
    if not os.path.exists("data/linear_landau"):
        os.makedirs("data/linear_landau")

    # save results
    np.save("data/linear_landau/sol_u_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), sol_midpoint_u)
    np.save("data/linear_landau/sol_t_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), setup.t_vec)

    # save parameters
    np.save("data/linear_landau/sol_setup_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), setup)
