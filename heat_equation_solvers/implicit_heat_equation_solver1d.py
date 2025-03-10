import numpy as np
import matplotlib.pyplot as plt
from heat_equation_solvers.heat_equation_solver1d import HeatEquationSolver1D
from scipy.linalg import solve_banded


class ImplicitHeatEquationSolver1D(HeatEquationSolver1D):
    def __init__(self, N, tau, t_max=5, v: float = 0.1, L: int = 1):
        """Initialize the ImplicitHeatEquationSolver object.
        This solver implements the implicit method for solving the heat equation with Dirichlet boundary conditions
        in 1D. The solver uses a finite difference discretization of the heat equation and implicit time stepping.
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
        super().__init__(N, tau, t_max, v, L)

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
        return (1 + 2 * self.alpha) * np.eye(self.N - 1) - self.alpha * (
            np.eye(self.N - 1, k=1) + np.eye(self.N - 1, k=-1)
        )

    def _solve(self):
        """
        Solves the heat equation implicitly using scipy's banded matrix solver.

        This method implements the core solving algorithm for the 1D heat equation using
        an implicit scheme. It uses scipy's solve_banded function to efficiently solve
        the tridiagonal system at each time step.

        Attributes modified:
            self.T: List of time points
            self.u: List of solution vectors at each time step
        """
        t = 0
        n = self.N - 1

        # Set up the tridiagonal matrix in banded form
        ab = np.zeros((3, n))
        ab[0, 1:] = -self.alpha  # upper diagonal
        ab[1, :] = 1 + 2 * self.alpha  # main diagonal
        ab[2, :-1] = -self.alpha  # lower diagonal

        while t < self.t_max:
            t += self.tau
            self.T.append(t)

            # Solve the system using scipy's banded matrix solver
            x = solve_banded((1, 1), ab, self.u[-1])
            self.u.append(x)
