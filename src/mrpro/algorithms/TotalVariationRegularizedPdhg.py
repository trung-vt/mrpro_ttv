"""Total Variation (TV)-Regularized PDHG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch

from mrpro.algorithms.VariationalRegularizedPdhg import VariationalRegularizedPdhg
from mrpro.operators import FiniteDifferenceOp, LinearOperator, LinearOperatorMatrix
from mrpro.operators.functionals import L1NormViewAsReal
from mrpro.utils import unsqueeze_right


class TotalVariationRegularizedPdhg(VariationalRegularizedPdhg):
    r"""TV-regularized PDHG.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \sum_i l_i ||\nabla_i x||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`l_i` are the strengths of the regularization
    along the different dimensions and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`.
    """

    def get_other_l1_terms(
        self,
        regularization_weight: Sequence[float] | Sequence[torch.Tensor],
        ndim: int,
    ) -> Tuple[L1NormViewAsReal,]:
        """Get the gradient term for the TV regularisation functional sum.

        Parameters
        ----------
        regularization_weight
            The regularization weights for the gradient term.
        ndim
            The number of dimensions to match the weight of the L1 norm to the measurement data.

        Returns
        -------
            The gradient term.
        """
        grad_term = L1NormViewAsReal(weight=unsqueeze_right(regularization_weight, ndim))
        return (grad_term,)

    def get_operator_matrix(
            self, acquisition_operator: LinearOperator | LinearOperatorMatrix, image_shape=None
    ) -> LinearOperatorMatrix:
        """Get the operator matrix for the TV regularisation functional sum.

        Parameters
        ----------
        acquisition_operator
            The acquisition operator.

        Returns
        -------
            The operator matrix for the TV regularisation functional sum. The matrix has the form
            :math:`\begin{pmatrix} A \\ \nabla_i \\end{pmatrix}` where :math:`A` is the acquisition operator
            and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
            different dimensions :math:`i`.
        """
        nabla_op = FiniteDifferenceOp(dim=self.regularization_dim, mode='forward')
        return LinearOperatorMatrix(((acquisition_operator,), (nabla_op,)))
