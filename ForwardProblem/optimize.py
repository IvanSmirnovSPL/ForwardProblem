from forward import calculate_forward_problem, residual
from wellborn import WellParams
from mass_flow import V
from numpy.typing import NDArray
from typing import List, Tuple
import numpy as np
from vizual import makeVizualData, show_gradients
import matplotlib.pyplot as plt
from wellborn import dims

from vizual import makeVizualData





def grad_calc(
        q: List[float],
        well_params: WellParams,
        F_0: float,
        J: int,
        v: V,
        P_ref: List[float],
        G_ref: List[float],
        weight: NDArray,
        grid_ref: NDArray,
        layers: List[float],
        delta=5e1) -> NDArray:
    """
    Calculate functional gradient.

    :param q:
    :param well_params:
    :param F_0:
    :param J:
    :param v:
    :param P_ref:
    :param G_ref:
    :param weight:
    :param grid:
    :param grid_ref:
    :param layers:
    :param delta:
    :return:
    """
    q = np.array(q)
    points = []
    functional = []
    for i in range(q.size):
        u = np.zeros(q.size)
        u[i] = delta
        points.append(q + u)
        points.append(q - u)
    for i in range(len(points)):
        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, points[i], v, None)
        debit, pressure = residual(P_ref, G_ref, P_sol, G_sol, J, grid_ref.size)
        F = debit * weight[0] + pressure * weight[1]
        functional.append(F)
    grad = np.zeros(q.size)
    for i in range(grad.size):
        grad[i] = (functional[i * 2] - functional[i * 2 + 1]) / (2 * delta)
    return np.array(grad) / np.linalg.norm(grad)


def minimize(
        q_cur: NDArray,
        q_ref: NDArray,
        well_params: WellParams,
        J: int,
        weight: NDArray,
        layers: NDArray,) -> Tuple:
    """
    Function to solve inverse problem.

    :param h: grid step
    :param layers: positions of inflows.
    :param v: velocity.
    :param weight: weights.
    :param grid: solver grid.
    :param grid_ref: reference data grid.
    :param P_ref: wellborn pressure reference.
    :param G_ref: debit reference.
    :param q_sol: initial guess.
    :return:
    """
    q = []
    functional = []

    P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_0, v, None)
    debit, pressure = residual(P_ref, G_ref, P_sol, G_sol, J, grid_ref.size)
    P_init, G_init = P_sol, G_sol
    F = debit * weight[0] + pressure * weight[1]
    F_0 = F
    print(f'F_0: {F_0}')
    q.append(q_cur)
    functional.append(F / F_0)

    k = 0
    while functional[-1] >= 1e-2 and k < 50:
        k += 1
        grad = grad_calc(q_cur, well_params, F_0, J, v, P_ref, G_ref, weight, grid_ref, layers)
        a = np.array(list(grad)) * 10
        q_cur = q_cur - np.array(list(grad)) * 10
        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_cur, v, None)
        debit, pressure = residual(P_ref, G_ref, P_sol, G_sol, J, grid_ref.size)
        F = debit * weight[0] + pressure * weight[1]
        q.append(q_cur)
        functional.append(F / F_0)
    return q_cur, functional, q, G_sol, P_sol, G_init, P_init


if __name__ == "__main__":
    well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
    J = 91
    weight = np.array([1e-4, 1e13])
    q_ref = np.array([60, 40]) * 10  # m^3 / day
    layers = np.array([3.5, 6.5])
    q_0 = np.array([80, 20]) * 10

    P_ref, G_ref, grid_ref, v, inflows_ref, grid = calculate_forward_problem(well_params, 91, layers, q_ref, None, q_ref)

    show_gradients(P_ref, G_ref, grid_ref, v, J, layers, well_params, 1, 40, np.linspace(70, 50, num=11))
    exit()


    q_cur, functional, q, G_sol, P_sol, G_init, P_init = minimize(q_0, None, well_params, J, weight, layers)

    makeVizualData(inflows_ref, well_params, grid_ref, grid, G_sol, G_ref, P_sol, P_ref, v, J, path='images_opt')

    fig, ax = plt.subplots(2, 2)
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]

    ax1.plot(G_sol, grid, label='G_sol', lw=9)
    ax1.plot(G_ref, grid_ref, label='G_ref', lw=5)
    ax1.plot(G_init, grid, label='G_init', lw=2)
    ax1.set_title('Mass flow')
    ax1.grid()
    ax1.invert_yaxis()
    ax1.legend()

    ax2.plot(P_sol, grid, label='P_sol', lw=9)
    ax2.plot(P_ref, grid_ref, label='P_ref', lw=5)
    ax2.plot(P_init, grid, label='P_init', lw=2)
    ax2.set_title('Wellborn pressure')
    ax2.grid()
    ax2.invert_yaxis()
    ax2.legend()

    for i in range(len(q[0])):
        q_i = [q[j][i] for j in range(len(q))]
        print(q_i)
        ax3.plot(q_i, '-o', label=f'$q_{i}$', lw=2)
        ax3.plot(np.ones(len(q)) * q_ref[i], label=f'$q_{i} ref$', lw=2)
    ax3.set_title('q')
    ax3.grid()
    ax3.legend()

    ax4.plot(np.log(functional), '-o', label='functional', lw=2)
    ax4.set_title('Wellbohr functional')
    ax4.grid()
    ax4.legend()

    plt.show()

    print(q_cur, q_ref, sep='\n')
    print('functional', functional)
