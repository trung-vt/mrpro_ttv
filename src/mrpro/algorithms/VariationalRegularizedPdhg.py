"""Variational-regularized PDHG (TV, TGV, etc.)."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.data.Dataclass import Dataclass
from mrpro.operators import LinearOperator, LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared
from mrpro.utils import normalize_index


class VariationalRegularizedPdhg:
    r"""Variational-regularized PDHG (TV, TGV, etc.).

    This algorithm solves the problem :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \mathcal{R}(x)`
    by using the PDHG-algorithm. :math:`y` is the acquired measurement data, :math:`\mathcal{R}` is the regularization
    with respect to :math:`x`.
    """

    max_iterations: int
    """Maximum number of PDHG iterations."""

    tolerance: float
    """Tolerance of PDHG for relative change of the primal solution."""

    regularization_dim: Sequence[int]
    """Dimensions along which the regularization is applied :math:`i`."""

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        tolerance: float = 0,
        regularization_dim: Sequence[int],
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        max_iterations
            Maximum number of PDHG iterations
        tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.
        regularization_dim
            Dimensions along which the total variation reguarization is applied (:math:`i`).

        Raises
        ------
        ValueError
            If `regularization_dim` contains repeated values.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')
        self.regularization_dim = regularization_dim

    def get_regularization_dim(self, measurement: torch.Tensor | Dataclass) -> Sequence[int]:
        regularization_dim = tuple(normalize_index(measurement.ndim, idx) for idx in self.regularization_dim)
        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')
        return regularization_dim

    def get_initial_value(self, initial_image: torch.Tensor) -> torch.Tensor:
        """Get the initial value for the variational regularized PDHG operator.

        Parameters
        ----------
        initial_image
            The initial image guess.

        Returns
        -------
            The initial value for the variational regularized PDHG operator.
        """
        return initial_image

    def process_pdhg_output(self, pdhg_output: torch.Tensor | Dataclass) -> torch.Tensor | Dataclass:
        return pdhg_output

    def __call__(
        self,
        acquisition_operator: LinearOperator,
        regularization_weight: Any,
        measurement: torch.Tensor | Dataclass,
        initial_image: torch.Tensor | Dataclass,
    ) -> torch.Tensor | Dataclass:
        """Call the reconstruction."""
        data_term = 0.5 * L2NormSquared(target=measurement.data)
        (pdhg_output,) = pdhg(
            f=ProximableFunctionalSeparableSum(
                data_term, **self.get_other_l1_terms(regularization_weight, measurement.data.ndim)
            ),
            g=None,
            operator=self.get_operator_matrix(acquisition_operator, initial_image.data.shape),
            initial_values=(self.get_initial_value(initial_image),),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        return self.process_pdhg_output(pdhg_output)

    @abstractmethod
    def get_other_l1_terms(
            self, regularization_weight: Any, ndim: int
    ) -> Sequence[L1NormViewAsReal]:
        """Get the other L1 terms for the variational regularisation functional sum."""

    @abstractmethod
    def get_operator_matrix(
            self, acquisition_operator: LinearOperator | LinearOperatorMatrix, image_shape: Sequence[int]
    ) -> LinearOperatorMatrix:
        """Get the operator matrix for the variational regularisation."""
