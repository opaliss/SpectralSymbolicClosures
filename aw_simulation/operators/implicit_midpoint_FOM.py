"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: March 5th, 2024
"""
import numpy as np
import scipy.optimize
import scipy
from operators.adaptive import check_if_update_needed, updated_u, updated_alpha, get_projection_matrix


def implicit_nonlinear_equation(y_new, y_old, dt, right_hand_side):
    """return the nonlinear equation for implicit midpoint to optimize.

    :param y_new: 1d array, y_{n+1}
    :param y_old: 1d array, y_{n}
    :param dt: float, time step t_{n+1} - t_{n}
    :param right_hand_side: function, a function of the rhs of the dynamical system dy/dt = rhs(y, t)
    :return: y_{n+1} - y_{n} -dt*rhs(y=(y_{n}+y_{n+1})/2, t=t_{n} + dt/2)
    """
    return y_new - y_old - dt * right_hand_side(y=0.5 * (y_old + y_new))


def implicit_midpoint_solver_FOM(y_0, right_hand_side, param, r_tol=1e-8, a_tol=1e-15, max_iter=100, adaptive=True):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param param: object of SimulationSetup with all the simulation setup parameters
    :param max_iter: maximum iterations of nonlinear solver, default is 100
    :param a_tol: absolute tolerance nonlinear solver, default is 1e-15
    :param r_tol: relative tolerance nonlinear solver, default is 1e-8
    :param y_0: initial condition
    :param adaptive: boolean
    :param right_hand_side: function of the right-hand-side, i.e. dy/dt = rhs(y, t)

    Returns
    -------
    u: (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # initialize the solution matrix
    y_sol = np.zeros((len(y_0), len(param.t_vec)), dtype="complex128")
    y_sol[:, 0] = y_0

    # for-loop each time-step
    for tt in range(1, len(param.t_vec)):
        # print out the current time stamp
        print("\n time = ", param.t_vec[tt])
        if adaptive:
            # updated u (electron 1) parameter
            u_e1_curr = updated_u(u_prev=param.u_e1[-1],
                                  alpha_prev=param.alpha_e1[-1],
                                  C00=y_sol[:, tt - 1][0],
                                  C10=y_sol[:, tt - 1][param.Nx_total])

            # updated alpha (electron 1) parameter
            alpha_e1_curr = updated_alpha(alpha_prev=param.alpha_e1[-1],
                                          C20=y_sol[:, tt - 1][2 * param.Nx_total],
                                          C10=y_sol[:, tt - 1][param.Nx_total],
                                          C00=y_sol[:, tt - 1][0])

            # update u (electron 2) parameter
            u_e2_curr = updated_u(u_prev=param.u_e2[-1], alpha_prev=param.alpha_e2[-1],
                                  C00=y_sol[:, tt - 1][param.Nv * param.Nx_total],
                                  C10=y_sol[:, tt - 1][param.Nv * param.Nx_total + param.Nx_total])

            # update alpha (electron 2) parameter
            alpha_e2_curr = updated_alpha(alpha_prev=param.alpha_e2[-1],
                                          C20=y_sol[:, tt - 1][param.Nv * param.Nx_total + 2 * param.Nx_total],
                                          C10=y_sol[:, tt - 1][param.Nv * param.Nx_total + param.Nx_total],
                                          C00=y_sol[:, tt - 1][param.Nv * param.Nx_total])

            # update u (ions) parameter
            u_i_curr = updated_u(u_prev=param.u_i[-1], alpha_prev=param.alpha_i[-1],
                                 C00=y_sol[:, tt - 1][2 * param.Nv * param.Nx_total],
                                 C10=y_sol[:, tt - 1][2 * param.Nv * param.Nx_total + param.Nx_total])

            # update alpha (ions) parameter
            alpha_i_curr = updated_alpha(alpha_prev=param.alpha_i[-1],
                                         C20=y_sol[:, tt - 1][2 * param.Nv * param.Nx_total + 2 * param.Nx_total],
                                         C10=y_sol[:, tt - 1][2 * param.Nv * param.Nx_total + param.Nx_total],
                                         C00=y_sol[:, tt - 1][2 * param.Nv * param.Nx_total])

            # electron 1 check mark
            if check_if_update_needed(u_s_curr=u_e1_curr, u_s=param.u_e1[-1], u_s_tol=param.u_tol,
                                      alpha_s_curr=alpha_e1_curr, alpha_s=param.alpha_e1[-1],
                                      alpha_s_tol=param.alpha_tol):
                print("updating u or alpha (electron 1)")
                print("ue1 = ", u_e1_curr)
                print("alpha_e1 = ", alpha_e1_curr)
                # get Hermite projection matrix
                P, case = get_projection_matrix(u_s_curr=u_e1_curr, u_s=param.u_e1[-1], alpha_s_curr=alpha_e1_curr,
                                                alpha_s=param.alpha_e1[-1], Nx_total=param.Nx_total, Nv=param.Nv,
                                                alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)
                if case == 1:
                    print("(e1) tolerance met for u and alpha")
                    # update parameters
                    param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)
                    param.replace_u_e1(u_e1_curr=u_e1_curr)

                elif case == 2:
                    print("(e1) tolerance met for u")
                    param.replace_u_e1(u_e1_curr=u_e1_curr)

                elif case == 3:
                    print("(e1) tolerance met for alpha")
                    param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)

                # project the previous timestamp results
                y_sol[:, tt - 1][:param.Nv * param.Nx_total] = P @ y_sol[:, tt - 1][:param.Nv * param.Nx_total]

            # electron 2 check mark
            if check_if_update_needed(u_s_curr=u_e2_curr, u_s=param.u_e2[-1], u_s_tol=param.u_tol,
                                      alpha_s_curr=alpha_e2_curr, alpha_s=param.alpha_e2[-1],
                                      alpha_s_tol=param.alpha_tol):
                print("updating u or alpha (electron 2)")
                print("ue2 = ", u_e2_curr)
                print("alpha_e2 = ", alpha_e2_curr)
                # get Hermite projection matrix
                P, case = get_projection_matrix(u_s_curr=u_e2_curr, u_s=param.u_e2[-1], alpha_s_curr=alpha_e2_curr,
                                                alpha_s=param.alpha_e2[-1], Nx_total=param.Nx_total, Nv=param.Nv,
                                                alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)
                if case == 1:
                    print("(e2) tolerance met for u and alpha")
                    # update parameters
                    param.replace_alpha_e2(alpha_e2_curr=alpha_e2_curr)
                    param.replace_u_e2(u_e2_curr=u_e2_curr)

                elif case == 2:
                    print("(e2) tolerance met for u")
                    param.replace_u_e2(u_e2_curr=u_e2_curr)

                elif case == 3:
                    print("(e2) tolerance met for alpha")
                    param.replace_alpha_e2(alpha_e2_curr=alpha_e2_curr)

                # project the previous timestamp results
                y_sol[:, tt - 1][param.Nv * param.Nx_total: 2 * param.Nv * param.Nx_total] = \
                    P @ y_sol[:, tt - 1][param.Nv * param.Nx_total: 2 * param.Nv * param.Nx_total]

            # ion check mark
            if check_if_update_needed(u_s_curr=u_i_curr, u_s=param.u_i[-1], u_s_tol=param.u_tol,
                                      alpha_s_curr=alpha_i_curr, alpha_s=param.alpha_i[-1],
                                      alpha_s_tol=param.alpha_tol):
                print("updating u or alpha (ions)")
                # get Hermite projection matrix
                P, case = get_projection_matrix(u_s_curr=u_i_curr, u_s=param.u_i[-1], alpha_s_curr=alpha_i_curr,
                                                alpha_s=param.alpha_i[-1], Nx_total=param.Nx_total, Nv=param.Nv,
                                                alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)

                if case == 1:
                    print("(ion) tolerance met for u and alpha")
                    # update parameters
                    param.replace_alpha_i(alpha_i_curr=alpha_i_curr)
                    param.replace_u_i(u_i_curr=u_i_curr)

                elif case == 2:
                    print("(ion) tolerance met for u")
                    param.replace_u_i(u_i_curr=u_i_curr)

                elif case == 3:
                    print("(ion) tolerance met for alpha")
                    param.replace_alpha_i(alpha_i_curr=alpha_i_curr)

                # project the previous timestamp results
                y_sol[:, tt - 1][2 * param.Nv * param.Nx_total: 3 * param.Nv * param.Nx_total] = \
                    P @ y_sol[:, tt - 1][2 * param.Nv * param.Nx_total: 3 * param.Nv * param.Nx_total]

            # update parameters electron 1
            param.add_alpha_e1(alpha_e1_curr=param.alpha_e1[-1])
            param.add_u_e1(u_e1_curr=param.u_e1[-1])

            # update parameters electron 2
            param.add_alpha_e2(alpha_e2_curr=param.alpha_e2[-1])
            param.add_u_e2(u_e2_curr=param.u_e2[-1])

            # update parameters ions
            param.add_alpha_i(alpha_i_curr=param.alpha_i[-1])
            param.add_u_i(u_i_curr=param.u_i[-1])

        y_sol[:, tt] = scipy.optimize.newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                                            y_old=y_sol[:, tt - 1],
                                                                                            right_hand_side=right_hand_side,
                                                                                            dt=param.dt),
                                                    xin=y_sol[:, tt - 1],
                                                    maxiter=max_iter,
                                                    method='gmres',
                                                    f_tol=a_tol,
                                                    f_rtol=r_tol,
                                                    verbose=True)
    return y_sol, param
