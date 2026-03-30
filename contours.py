import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from DPHI import D_PHI


def Contour_DPHI():
    # A 2D example of contour plots of D-PHI values, corresponds to the example shown in Figure 3

    # The same aspiration level (ai) and reservation level (ri) are used for both objectives
    # That is, a = (5, 5), r = (30, 30)
    ai = 5
    ri = 30
    epsilon_r = 0.2

    # Define the sampling range of the objective space
    y1 = np.linspace(-40, 60, 1000)  # Sampling points along y1 axis
    y2 = np.linspace(-40, 60, 1000)  # Sampling points along y2 axis

    # Generate a structured grid
    Y1, Y2 = np.meshgrid(y1, y2)

    # Reference point in the desirability space (same epsilon_r for all objectives)
    ref_point = np.array([-epsilon_r,-epsilon_r])

    # Compute desirability function values for each objective over the grid
    df_cal1 = D_PHI(Y1, ai,ri)
    D1 = df_cal1.df()  # Desirability function value of Y1
    df_cal2 = D_PHI(Y2, ai, ri)
    D2 = df_cal2.df()  # Desirability function value of Y2

    # The difference between desirability function values and the lower limit (-epsilon_r)
    diff1 = D1-ref_point[0]
    diff2 = D2-ref_point[1]

    # Compute D-PHI values on the grid: (D1 - (-epsilon_r)) * (D2 - (-epsilon_r))
    Z = diff1 * diff2


    # --- Contour plot in the objective space (y1, y2) ---
    fig1 = plt.figure()
    fig1.set_dpi(300)

    # Contour plot based on structured grid sampling of Z = f(y1, y2)
    plt.contourf(Y1, Y2, Z, levels=50, cmap='viridis_r')
    plt.colorbar()  # add colorbar

    # Draw a rectangle representing the region between aspiration and reservation levels
    rect = mpatches.Rectangle((ai, ai), ri-ai, ri-ai, fill=None, edgecolor='darkgray', linewidth=2)
    plt.gca().add_patch(rect)

    # Plot the aspiration point a
    plt.scatter(ai, ai, color='red', marker='s')
    # Plot the reservation point r
    plt.scatter(ri, ri, color='black', marker='s')

    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.tight_layout()


    # --- Contour plot in the desirability space (D1, D2) ---
    fig2 = plt.figure()
    fig2.set_dpi(300)
    # Here D1 and D2 are obtained by mapping Y1 and Y2 through the desirability function.
    plt.contourf(D1, D2, Z, levels=50, cmap='viridis_r')
    plt.colorbar()  # add colorbar

    # In desirability space: D(ai) = 1, D(ri) = 0
    rect = mpatches.Rectangle((0, 0), 1, 1, fill=None, edgecolor='darkgray', linewidth=2)
    plt.gca().add_patch(rect)

    # Plot the aspiration point in the desirability space
    plt.scatter(1, 1, color='red', marker='s')
    # Plot the reservation point in the desirability space
    plt.scatter(0, 0, color='black', marker='s')

    plt.xlabel(r'$D_1(y_1)$')
    plt.ylabel(r'$D_2(y_2)$')

    # Lower bound: D(ri) - epsilon_r = -0.2
    # Upper bound: D(ai) + epsilon_r = 1.2
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    Contour_DPHI()