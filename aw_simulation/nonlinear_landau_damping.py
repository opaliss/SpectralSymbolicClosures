"""Module to run the nonlinear Landau damping testcase

Author: Opal Issan
Date: Oct 18th, 2024
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
    setup = SimulationSetupFOM(Nx=100,
                               Nx_total=201,
                               Nv=300,
                               epsilon=0.5,
                               alpha_e=np.sqrt(2),
                               alpha_i=np.sqrt(2 / 1836),
                               u_e=0,
                               u_i=0,
                               L=4 * np.pi,
                               dt=0.01,
                               T0=0,
                               T=100,
                               nu=0,
                               hyper_rate=3,
                               col_type="hyper",
                               closure_type="truncation")

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.Nv * setup.Nx_total, dtype="complex128")
    # electron equilibrium
    y0[setup.Nx] = 1 / setup.alpha_e
    # electron perturbation
    y0[setup.Nx + 1] = 0.5 * setup.epsilon / setup.alpha_e
    y0[setup.Nx - 1] = 0.5 * setup.epsilon / setup.alpha_e

    # ions (unperturbed)
    C0_ions = np.zeros(setup.Nx_total)
    C0_ions[setup.Nx] = 1 / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_FOM(y_0=y0,
                                                         right_hand_side=rhs,
                                                         r_tol=1e-10,
                                                         a_tol=1e-12,
                                                         max_iter=100,
                                                         param=setup,
                                                         adaptive=False)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # make directory
    if not os.path.exists("data/nonlinear_landau"):
        os.makedirs("data/nonlinear_landau")

    # save results every n steps
    skip = 10

    # save results
    np.save("data/nonlinear_landau/sol_u_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), sol_midpoint_u[:, ::skip])
    np.save("data/nonlinear_landau/sol_t_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), setup.t_vec[::skip])

    # save parameters
    np.save("data/nonlinear_landau/sol_setup_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type) + "_" + str(setup.hyper_rate) + "_nu_" + str(setup.nu), setup)
