"""Total Generalized Variation (TGV) Denoising using PDHG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from mrpro.algorithms.TotalGeneralizedVariationRegularizedPdhg import (
    TotalGeneralizedVariationRegularizedPdhg,
    TgvOperatorMatrix
)
from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.data.IData import IData
from mrpro.operators import ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L2NormSquared
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.utils import normalize_index


@overload
def total_generalized_variation_denoising(
    idata: IData,
    regularization_dim: Sequence[int],
    regularization_weight_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    regularization_weight_sym_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> IData: ...


@overload
def total_generalized_variation_denoising(
    idata: torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    regularization_weight_sym_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> torch.Tensor: ...


def check_regularization_dim(
    regularization_dim: Sequence[int],
    regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
    ndim: int
):
    if len(regularization_dim) != len(regularization_weight):
        raise ValueError('Regularization dimensions and weights must have the same length')
    regularization_dim = tuple(normalize_index(ndim, idx) for idx in regularization_dim)
    if len(regularization_dim) != len(set(regularization_dim)):
        raise ValueError('Repeated values are not allowed in regularization_dim')
    return regularization_dim


def prepare_regularization_weight(
        regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
        regularization_dim: Sequence[int]
):
    return torch.as_tensor(
        regularization_weight
        if isinstance(regularization_weight, Sequence)
        else [regularization_weight] * len(regularization_dim)
    )


def total_generalized_variation_denoising(
    idata: IData | torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    regularization_weight_sym_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> IData | torch.Tensor:
    r"""Apply total generalized variation denoising.

    This algorithm solves the problem
    :math:`min_x \frac{1}{2}||x - y||_2^2 + \\sum_i l_i ||\nabla_i x||_1 + \\sum_i ||\mathcal{E}_i v||_1`
    by using the PDHG-algorithm. :math:`y` is the given noisy image, :math:`l_i` are the strengths of the regularization
    along the different dimensions,
    :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`,
    and :math:`\mathcal{E}_i` is the symmetrized gradient operator applied to :math:`v`
    along the same dimensions.

    Parameters
    ----------
    idata
        input image
    regularization_dim
        Dimensions along which the total variation reguarization is applied (:math:`i`).
    regularization_weight
        Strengths of the regularization (:math:`l_i`). If a single values is given, it is applied to all dimensions.
        If a sequence is given, it must have the same length as `regularization_dim`.
    initial_image
        Initial image. If `None` then the target image :math:`y` will be used.
    max_iterations
        Maximum number of PDHG iterations.
    tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.

    Returns
    -------
        The denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data
    initial_image = initial_image if initial_image is not None else img_tensor

    regularization_weight_grad_term_ = prepare_regularization_weight(
        regularization_weight_grad_term, regularization_dim
    )
    regularization_weight_sym_grad_term_ = prepare_regularization_weight(
        regularization_weight_sym_grad_term, regularization_dim
    )
    regularization_dim = check_regularization_dim(
        regularization_dim, regularization_weight_grad_term_, img_tensor.ndim
    )

    (img_tensor,) = pdhg(
        f=ProximableFunctionalSeparableSum(
            L2NormSquared(target=img_tensor),
            *TotalGeneralizedVariationRegularizedPdhg.get_l1_terms(
                (regularization_weight_grad_term_, regularization_weight_sym_grad_term_)
            ),
        ),
        g=None,
        operator=TgvOperatorMatrix(
            regularization_dim,
            acquisition_operator=IdentityOp(),
            image_shape=initial_image.data.shape
        ),
        initial_values=(initial_image,),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
