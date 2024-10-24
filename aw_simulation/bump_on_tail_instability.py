"""Module to run bump-on-tail instability testcase

Author: Opal Issan
Date: Oct 2nd, 2024
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.FOM import nonlinear_full, charge_density_two_stream
from operators.implicit_midpoint_FOM import implicit_midpoint_solver_FOM
from operators.setup_FOM_two_stream import SimulationSetupTwoStreamFOM
import time
import numpy as np


def rhs(y):
    # initialize the rhs dy/dt
    dydt_ = np.zeros(len(y), dtype="complex128")

    # electric field computed
    E = setup.D_inv @ charge_density_two_stream(C0_electron_1=y[:setup.Nx_total],
                                                C0_electron_2=y[setup.Nx_total * setup.Nv: setup.Nx_total * (setup.Nv + 1)],
                                                C0_ions=C0_ions,
                                                alpha_e1=setup.alpha_e1,
                                                alpha_e2=setup.alpha_e2,
                                                alpha_i=setup.alpha_i,
                                                q_e1=setup.q_e1,
                                                q_e2=setup.q_e2,
                                                q_i=setup.q_i)

    # evolving electrons + ions
    # electron species (1) => bulk
    dydt_[: setup.Nv * setup.Nx_total] = setup.A_e1 @ y[: setup.Nv * setup.Nx_total] \
        + setup.B_e1 @ nonlinear_full(E=E, psi=y[:setup.Nv * setup.Nx_total], Nv=setup.Nv, Nx_total=setup.Nx_total)

    # electron species (2) => bump
    dydt_[setup.Nv * setup.Nx_total: 2 * setup.Nv * setup.Nx_total] = setup.A_e2 @ y[setup.Nv * setup.Nx_total: 2 * setup.Nv * setup.Nx_total] \
        + setup.B_e2 @ nonlinear_full(E=E, psi=y[setup.Nv * setup.Nx_total: 2 * setup.Nv * setup.Nx_total], Nv=setup.Nv, Nx_total=setup.Nx_total)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupTwoStreamFOM(Nx=60,
                                        Nx_total=121,
                                        Nv=350,
                                        epsilon=0.03,
                                        alpha_e1=np.sqrt(2),
                                        alpha_e2=1,
                                        alpha_i=np.sqrt(2 / 1836),
                                        u_e1=0,
                                        u_e2=4.5,
                                        u_i=0,
                                        L=20 * np.pi / 3,
                                        dt=1e-2,
                                        T0=0,
                                        T=25,
                                        nu=12,
                                        n0_e1=0.8,
                                        n0_e2=0.2,
                                        closure_type="truncation",
                                        col_type="hyper",
                                        hyper_rate=1)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(2 * setup.Nv * setup.Nx_total, dtype="complex128")
    # first electron 1 species (perturbed)
    y0[setup.Nx] = setup.n0_e1 / setup.alpha_e1
    y0[setup.Nx + 1] = 0.5 * setup.n0_e1 * setup.epsilon / setup.alpha_e1
    y0[setup.Nx - 1] = 0.5 * setup.n0_e1 * setup.epsilon / setup.alpha_e1
    # second electron species (perturbed)
    y0[setup.Nv * setup.Nx_total + setup.Nx] = setup.n0_e2 / setup.alpha_e2
    y0[setup.Nv * setup.Nx_total + setup.Nx + 1] = 0.5 * setup.n0_e2 * setup.epsilon / setup.alpha_e2
    y0[setup.Nv * setup.Nx_total + setup.Nx - 1] = 0.5 * setup.n0_e2 * setup.epsilon / setup.alpha_e2
    # ions (unperturbed + static)
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

    # make directory
    if not os.path.exists("data/bump_on_tail"):
        os.makedirs("data/bump_on_tail")

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save results
    np.save("data/bump_on_tail/sol_u_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), sol_midpoint_u)
    np.save("data/bump_on_tail/sol_t_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), setup.t_vec)

    # save parameters
    np.save("data/bump_on_tail/sol_setup_" + str(setup.Nv) + "_closure_" + str(setup.closure_type) + "_collisions_" + str(setup.col_type), setup)

