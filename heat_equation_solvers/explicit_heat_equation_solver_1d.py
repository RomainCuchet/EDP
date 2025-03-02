import numpy as np
import matplotlib.pyplot as plt


class ExplicitHeatEquationSolver1D:
    def __init__(self, N, tau, t_max=5, v: float = 0.1, L: int = 1):
        """Initialize the ExplicitHeatEquationSolver object.
        This solver implements the explicit method for solving the heat equation with Dirichlet boundary conditions
        in 1D. The solver uses a finite difference discretization of the heat equation and explicit time stepping.
        Initial conditions are set to a sine function : sin(pi*L*x) for x in [0,1]. L is set to the closest positive integer to preserve 0 condition at both boundaries
        Parameters
        ----------
        N : int
            Number of spatial discretization points.
        tau : float
            Time step size.
        t_max : float, optional
            Maximum simulation time (default is 5).

        v : float, optional
            Thermal diffusivity coefficient (default is 0.1).
        Attributes

        L : int
            Number of half-wavelengths in the initial condition.
        ----------
        h : float
            Spatial step size, calculated as 1/N.
        alpha : float
            Stability parameter, calculated as v*tau/h^2.
        X : ndarray
            Spatial grid points from 0 to 1.
        Raises
        ------
        ValueError
            If alpha > 1/2, which violates the stability condition.
        """
        self.N = N
        self.tau = tau
        self.t_max = t_max
        self.v = v
        self.h = 1 / N
        self.L = int(np.abs(L) + 0.5)
        self.alpha = self.v * self.tau / self.h**2

        if self.alpha > 1 / 2:
            raise ValueError("alpha must be below 1/2")

        self.X = np.linspace(0, 1, N + 1)
        self._solve()

    def _construct_matrix(self):
        """
        Constructs the matrix used in the explicit scheme for solving the heat equation.

        The matrix is a tridiagonal matrix of size (N-1)x(N-1) where:
        - Main diagonal elements are (1 - 2α)
        - Upper and lower diagonal elements are α
        - α is the stability parameter (dt/(dx)²)

        Returns:
            numpy.ndarray: A (N-1)x(N-1) tridiagonal matrix used for the explicit scheme calculation
        """
        return (1 - 2 * self.alpha) * np.eye(self.N - 1) + self.alpha * (
            np.eye(self.N - 1, k=1) + np.eye(self.N - 1, k=-1)
        )

    def _solve(self):
        t = 0
        self.T = [t]
        # TODO: Add parameter to allow custom initial conditions, implies parameter validation
        self.u0 = np.sin(
            np.pi * self.X[1 : self.N] * self.L
        )  # Initial condition, ensures 0 at boundaries
        self.u = [self.u0]

        M = self._construct_matrix()

        while t < self.t_max:
            t += self.tau
            self.T.append(t)
            self.u.append(np.dot(M, self.u[-1]))

    def display_solution(self, times_to_plot: list[float] | float):
        """
        Display the solution of the heat equation at specified time points.
        This method creates a plot showing the temperature distribution along the spatial domain
        at one or multiple time points.
        Parameters
        ----------
        times_to_plot : Union[list[float], float]
            Time point(s) at which to display the solution. Can be a single float value
            or a list of float values representing the times in seconds.
        Returns
        -------
        None
            Displays a matplotlib figure showing the temperature distribution.
        Notes
        -----
        - The solution is plotted as temperature (u) vs. position (X)
        - Each time point is represented as a separate line on the plot
        - A legend identifies each line with its corresponding time
        - The grid is enabled for better readability
        - The legend is positioned outside the plot area to avoid overlap
        """

        def get_time_index(t):
            return int(t / self.tau)

        if not isinstance(times_to_plot, list):
            times_to_plot = [times_to_plot]

        plt.figure()

        for time_to_plot in times_to_plot:
            time_to_plot = get_time_index(time_to_plot)
            if time_to_plot < len(self.u):
                plt.plot(
                    self.X[1 : self.N],
                    self.u[time_to_plot],
                    label=f"t={self.T[time_to_plot]:.2f} s",
                )

        plt.xlabel("X")
        plt.ylabel("u")
        plt.title("Heat Equation Solution - 1D")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()  # Adjusts the plot to ensure the legend fits
        plt.show()
