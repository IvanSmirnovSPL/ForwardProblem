import matplotlib.pyplot as plt
import numpy as np

from mass_flow import calc_velocity
from forward import calculate_forward_problem, residual


def makeVizualData(inflows_ref, well_params, grid_ref, grid, G_sol, G_ref, P_sol, P_ref, v, J, path='images'):
    fig, ax = plt.subplots(1, 3)
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    k = int((grid_ref.size - 1) / (grid.size - 1))
    ax1.scatter(np.array(G_sol) / 24, grid, label='$G_{sol}$', color='b', lw=4)
    ax1.plot(np.array(G_ref) / 24, grid_ref, label='$G_{ref}$', color='r', lw=2)
    foo = np.linspace(min(np.array(G_sol) / 24), max(np.array(G_sol) / 24), num=5)
    ax1.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$\frac{m^3}{hour}$"])
    ax1.set_title('Mass flow')
    ax1.grid()
    ax1.invert_yaxis()
    ax1.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax1.transAxes)
    ax1.legend()

    ax2.scatter([(p - 1) * 1e7 for p in P_sol], grid, label='$P_{sol} - 1 bar$', color='b', lw=4)
    ax2.plot([(p - 1) * 1e7 for p in P_ref], grid_ref, label='$P_{ref} - 1 bar$', color='r', lw=2)
    foo = np.linspace(min([(p - 1) * 1e7 for p in P_sol]), max([(p - 1) * 1e7 for p in P_sol]), num=4)
    ax2.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$10^{-7} bar$"])
    ax2.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax2.transAxes)
    ax2.set_title('Wellbohr pressure')
    ax2.grid()
    ax2.invert_yaxis()
    ax2.legend()

    def analytic_velocity(z):
        return calc_velocity(inflows_ref, z, well_params)

    a = list(map(analytic_velocity, grid_ref))
    ax3.plot(list(map(v, grid_ref)), grid_ref, color='b',
             label='spline ' + f'$ \Delta v{"%.2e" % abs(max(a) - min(a))}$', lw=4)

    ax3.plot(list(map(analytic_velocity, grid_ref)), grid_ref, color='r', label='analytic', lw=2)
    foo = np.linspace(min(a), max(a), num=4)
    ax3.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$\frac{m}{sec}$"])
    ax3.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax3.transAxes)
    ax3.set_title('Velocity')
    ax3.grid()
    ax3.invert_yaxis()
    ax3.legend()

    plt.savefig(f'{path}\sol{J}.png', dpi=1000)


def show_gradients(P_ref, G_ref, grid_ref, v, J, layers, well_params, num, fixed, space):
    d = []
    p = []
    q_ = []
    for q in space:
        if num == 1:
            q = np.array([q, fixed]) * 10
        else:
            q = np.array([fixed, q]) * 10

        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q, v, None)
        debit, pressure = residual(P_ref, G_ref, P_sol, G_sol, J, grid_ref.size)
        d.append(debit)
        p.append(pressure)
        q_.append(q[num - 1])
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(q_, d,  '-o', color='r', label='debit')
    for q, r in zip(q_, d):
        def vec(r):
            return np.array([r, 1])
        p1 = np.array([q, r])
        p2 = (vec(r) / np.linalg.norm(vec(r)) + p1 / np.linalg.norm(p1)) * 10 + p1

        print(p1, p2)
        ax[0].plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), '-o')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(q_, p, '-o', label='pressure')
    ax[1].legend()
    ax[1].grid(True)
    plt.show()