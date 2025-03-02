import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class HeatEquationSolver1D(ABC):
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

        self.T = [0]
        self.X = np.linspace(0, 1, N + 1)

        # TODO: Add parameter to allow custom initial conditions, implies parameter validation
        self.u0 = np.sin(
            np.pi * self.X[1 : self.N] * self.L
        )  # Initial condition, ensures 0 at boundaries
        self.u = [self.u0]

        self.M = self._construct_matrix()
        self._solve()

    @abstractmethod
    def _construct_matrix(self):
        pass

    @abstractmethod
    def _solve(self):
        pass

    def get_solution_plot(
        self, times_to_plot: list[float] | float, show_plot=True, ax=None
    ):
        """
        Display the solution of the heat equation at specified time points.
        This method creates a plot showing the temperature distribution along the spatial domain
        at one or multiple time points.
        Parameters
        ----------
        times_to_plot : Union[list[float], float]
            Time point(s) at which to display the solution. Can be a single float value
            or a list of float values representing the times in seconds.
        show_plot : bool, optional
            Whether to display the plot (default is True).
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on (default is None).
        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """

        def get_time_index(t):
            return int(t / self.tau)

        if not isinstance(times_to_plot, list):
            times_to_plot = [times_to_plot]

        if ax is None:
            fig, ax = plt.subplots()

        for time_to_plot in times_to_plot:
            time_to_plot = get_time_index(time_to_plot)
            if time_to_plot < len(self.u):
                ax.plot(
                    self.X[1 : self.N],
                    self.u[time_to_plot],
                    label=f"t={self.T[time_to_plot]:.2f} s",
                )

        ax.set_xlabel("X")
        ax.set_ylabel("u")
        ax.set_title("Heat Equation Solution - 1D")
        ax.grid()
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if show_plot:
            plt.show()

        return ax
