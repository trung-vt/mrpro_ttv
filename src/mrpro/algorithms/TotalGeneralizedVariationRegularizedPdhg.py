"""Total Generalized Variation (TGV)-Regularized PDHG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.algorithms.VariationalRegularizedPdhg import VariationalRegularizedPdhg
from mrpro.operators import FiniteDifferenceOp, LinearOperator, LinearOperatorMatrix, SymmetrizedGradientOp
from mrpro.operators.functionals import L1NormViewAsReal


class TotalGeneralizedVariationRegularizedPdhg(VariationalRegularizedPdhg):
    r"""TGV-regularized PDHG.

    This algorithm solves the problem
    :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \sum_i l_i ||\nabla_i x - v||_1 + \sum_i ||\mathcal{E}_i v||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`l_i` are the strengths of the regularization
    along the different dimensions, :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`, and :math:`\mathcal{E}_i` is the symmetrized gradient operator applied to :math:`v`
    along the same dimensions.
    """

    def get_initial_value(self, initial_image: torch.Tensor) -> torch.Tensor:
        """Get the initial value for the TGV PDHG operator.

        Parameters
        ----------
        initial_image
            The initial image guess.

        Returns
        -------
            The initial value for the TGV PDHG operator. The first element is the initial image, and the
            remaining :math:`k` elements are zero tensors (:math:`k` being the number of regularization dimensions).
        """
        return torch.cat(
            (initial_image.unsqueeze(0), initial_image.new_zeros((len(self.regularization_dim), *initial_image.shape))),
            dim=0,
        )

    def process_pdhg_output(self, pdhg_output):
        return pdhg_output[0]

    @classmethod
    def get_l1_terms(
        cls, regularization_weight: Sequence[float] | Sequence[Sequence[float]] | Sequence[Sequence[torch.Tensor]]
    ) -> tuple[L1NormViewAsReal, L1NormViewAsReal]:
        """Get the gradient term and the symmetrized gradient term for the TGV regularisation functional sum.

        Parameters
        ----------
        regularization_weight
            The regularization weights for the gradient term and the symmetrized gradient term.
        ndim
            The number of dimensions to match the weight of the L1 norm to the measurement data.

        Returns
        -------
            The gradient term and the symmetrized gradient term.
        """
        grad_weight, sym_grad_weight = regularization_weight
        grad_term = L1NormViewAsReal(weight=grad_weight)
        sym_grad_term = L1NormViewAsReal(weight=sym_grad_weight)
        return (grad_term, sym_grad_term)

    def get_operator_matrix(
        self, acquisition_operator: LinearOperator | LinearOperatorMatrix, image_shape: Sequence[int]
    ) -> TgvOperatorMatrix:
        """Get the operator matrix for the TGV PDHG operator.

        Parameters
        ----------
        acquisition_operator
            The acquisition operator.
        image_shape
            The shape of the image data to reconstruct.

        Returns
        -------
        TgvOperatorMatrix
            The operator matrix for the TGV PDHG operator.
        """
        return TgvOperatorMatrix(self.regularization_dim, acquisition_operator, image_shape)


class TgvOperatorMatrix(LinearOperatorMatrix):
    """Operator matrix for the TGV PDHG operator.

    This operator matrix combines the data term, gradient term, and symmetrized gradient term.

    The matrix has the form
    :math:`\begin{pmatrix} A \\ \nabla_i \\ \\mathcal{E}_i \\end{pmatrix}`
    where :math:`A` is the acquisition operator,
    :math:`\nabla_i` is the finite difference operator
    applied to :math:`x` along different dimensions :math:`i`,
    and :math:`\\mathcal{E}_i` is the symmetrized gradient operator applied to :math:`v` along
    the same dimensions :math:`i`.
    """

    def __init__(
            self,
            regularization_dim: Sequence[int],
            acquisition_operator: LinearOperator | LinearOperatorMatrix,
            image_shape: Sequence[int]
    ):
        """Initialize the TGV operator matrix."""
        v_shape = (len(regularization_dim), *image_shape)
        super().__init__(
            (
                (DataTermOperatorMatrixRow(acquisition_operator, v_shape),),
                (GradientTermOperatorMatrixRow(regularization_dim, mode='forward'),),
                (SymmetrizedGradientTermOperatorMatrixRow(image_shape, regularization_dim, mode='backward'),),
            )
        )


class DataTermOperatorMatrixRow(LinearOperator):
    """Operator matrix row corresponding to the data term (first row) for the TGV PDHG operator."""

    def __init__(self, acquisition_operator: LinearOperator, v_shape: Sequence[int]):
        """Initialize the operator matrix row."""
        super().__init__()
        self.acquisition_operator = acquisition_operator
        self.v_shape = v_shape

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator."""
        return self.acquisition_operator.forward(y[0])

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor]:
        """Adjoint operator."""
        return (torch.cat((self.acquisition_operator.adjoint(s)[0].unsqueeze(0), torch.zeros(self.v_shape)), dim=0),)


class GradientTermOperatorMatrixRow(LinearOperator):
    """Operator matrix row corresponding to the gradient term (second row) for the TGV PDHG operator."""

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ):
        """Initialize the operator matrix row."""
        super().__init__()
        # TODO: Normalize dim
        self.finite_difference_op = FiniteDifferenceOp(
            dim=dim,
            mode=mode,
            pad_mode=pad_mode,
        )

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator."""
        # (N1, N2, ... N_d) --> (k, N1, N2, ..., N_d)
        nabla_x = self.finite_difference_op.forward(y[0])[0]
        return (nabla_x - y[1:],)

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator."""
        adjoint_v = self.finite_difference_op.adjoint(s)[0]
        return (torch.cat((adjoint_v.unsqueeze(0), -s), dim=0),)


class SymmetrizedGradientTermOperatorMatrixRow(LinearOperator):
    """Operator matrix row corresponding to the symmetrized gradient term (third row) for the TGV PDHG operator."""

    def __init__(
        self,
        x_shape: Sequence[int],
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ):
        """Initialize the operator matrix row."""
        super().__init__()
        self.x_shape = x_shape
        self.symmetrized_gradient_op = SymmetrizedGradientOp(
            dim=dim,
            mode=mode,
            pad_mode=pad_mode,
        )

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator."""
        e_h_v = self.symmetrized_gradient_op.forward(y[1:])[0]
        return (e_h_v,)

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator."""
        adjoint_w = self.symmetrized_gradient_op.adjoint(s)[0]
        return (torch.cat((torch.zeros(self.x_shape).unsqueeze(0), adjoint_w), dim=0),)
