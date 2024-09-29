"""Module to run the adaptive (or non-adaptive) Hermite linear Landau damping full-order model (FOM) testcase

Author: Opal Issan
Date: May 10th, 2024
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
    for alpha_e in [1.5]:
        setup = SimulationSetupFOM(Nx=20,
                                   Nx_total=41,
                                   Nv=100,
                                   epsilon=1e-2,
                                   alpha_e=alpha_e,
                                   alpha_i=np.sqrt(2 / 1836),
                                   u_e=0,
                                   u_i=0,
                                   L=2 * np.pi,
                                   dt=1e-2,
                                   T0=0,
                                   T=30,
                                   nu=8)

        # initial condition: read in result from previous simulation
        y0 = np.zeros(setup.Nv * setup.Nx_total, dtype="complex128")
        # first electron 1 species (perturbed)
        y0[setup.Nx] = 1 / setup.alpha_e
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
        if not os.path.exists("../data/FOM/linear_landau/sample_" + str(setup.alpha_e)):
            os.makedirs("../data/FOM/linear_landau/sample_" + str(setup.alpha_e))

        # save the runtime
        np.save("../data/FOM/linear_landau/sample_" + str(setup.alpha_e) + "/sol_FOM_u_" + str(setup.Nv) + "_alpha_" + str(setup.alpha_e) + "_runtime_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

        # save results
        np.save("../data/FOM/linear_landau/sample_" + str(setup.alpha_e) + "/sol_FOM_u_" + str(setup.Nv) + "_alpha_" + str(setup.alpha_e) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
        np.save("../data/FOM/linear_landau/sample_" + str(setup.alpha_e) + "/sol_FOM_t_" + str(setup.Nv) + "_alpha_" + str(setup.alpha_e) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

        # save parameters
        np.save("../data/FOM/linear_landau/sample_" + str(setup.alpha_e) + "/sol_FOM_setup_" + str(setup.Nv) + "_alpha_" + str(setup.alpha_e) + "_" + str(setup.T0) + "_" + str(setup.T), setup)
