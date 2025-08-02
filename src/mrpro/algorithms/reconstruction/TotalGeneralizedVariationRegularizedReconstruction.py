"""Total Generalized Variation (TGV)-Regularized Reconstruction using PDHG."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from mrpro.algorithms.TotalGeneralizedVariationRegularizedPdhg import (
    TotalGeneralizedVariationRegularizedPdhg,
    TgvOperatorMatrix
)
from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators import ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L2NormSquared
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils import normalize_index


class TotalGeneralizedVariationRegularizedReconstruction(DirectReconstruction):
    r"""TGV-regularized reconstruction.

    This algorithm solves the problem
    :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \sum_i l_i ||\nabla_i x||_1 + \sum_i ||\mathcal{E}_i v||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`l_i` are the strengths of the regularization
    along the different dimensions,
    :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`,
    and :math:`\mathcal{E}_i` is the symmetrized gradient operator applied to :math:`v`
    along the same dimensions.
    """

    max_iterations: int
    """Maximum number of PDHG iterations."""

    tolerance: float
    """Tolerance of PDHG for relative change of the primal solution."""

    regularization_dim: Sequence[int]
    """Dimensions along which the total variation reguarization is applied :math:`i`."""

    regularization_weight_grad_term: torch.Tensor
    """Strengths of the regularization of the TGV PDHG operator's gradient term
    along different dimensions :math:`l_i`."""

    regularization_weight_sym_grad_term: torch.Tensor
    """Strengths of the regularization of the TGV PDHG operator's symmetrized gradient term
    along different dimensions :math:`l_i`."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
        *,
        max_iterations: int = 100,
        tolerance: float = 0,
        regularization_dim: Sequence[int],
        regularization_weight_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
        regularization_weight_sym_grad_term: float | Sequence[float] | Sequence[torch.Tensor],
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        kdata
            KData. If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are
            estimated based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mrpro.operators.FourierOp` used for reconstruction. If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mrpro.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the `Callable`. For this, `kdata` needs also to be provided.
            For examples have a look at the `mrpro.data.CsmData` class e.g. `~mrpro.data.CsmData.from_idata_walsh`
            or `~mrpro.data.CsmData.from_idata_inati`.
        noise
            KNoise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`. The `dcf` is only used to
            calculate a starting estimate for PDHG.
        max_iterations
            Maximum number of PDHG iterations
        tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.
        regularization_dim
            Dimensions along which the total variation reguarization is applied (:math:`i`).
        regularization_weight_grad_term
            Strengths of the regularization of the TGV PDHG operator's gradient term (:math:`l_i`).
            If a single values is given, it is applied to all dimensions.
            If a sequence is given, it must have the same length as `regularization_dim`.
        regularization_weight_sym_grad_term
            Strengths of the regularization of the TGV PDHG operator's symmetrized gradient term (:math:`l_i`).
            If a single values is given, it is applied to all dimensions.
            If a sequence is given, it must have the same length as `regularization_dim`.

        Raises
        ------
        ValueError
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is `None`.
        ValueError
            If `regularization_dim` contains repeated values.
        ValueError
            If the length of `regularization_dim` and `regularization_weight_grad_term` do not match
        ValueError
            If the length of `regularization_dim` and `regularization_weight_sym_grad_term` do not match
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')
        self.regularization_dim = regularization_dim

        if isinstance(regularization_weight_grad_term, float):
            regularization_weight_grad_term = [regularization_weight_grad_term] * len(regularization_dim)
        if len(regularization_dim) != len(regularization_weight_grad_term):
            raise ValueError('Regularization dimensions and weights must have the same length')
        self.regularization_weight_grad_term = torch.as_tensor(regularization_weight_grad_term)

        if isinstance(regularization_weight_sym_grad_term, float):
            regularization_weight_sym_grad_term = [regularization_weight_sym_grad_term] * len(regularization_dim)
        if len(regularization_dim) != len(regularization_weight_sym_grad_term):
            raise ValueError('Regularization dimensions and weights must have the same length')
        self.regularization_weight_sym_grad_term = torch.as_tensor(regularization_weight_sym_grad_term)

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            K-space data to reconstruct.

        Returns
        -------
            The reconstructed image.
        """
        regularization_dim = tuple(normalize_index(kdata.ndim, idx) for idx in self.regularization_dim)
        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')

        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)

        acquisition_operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op

        initial_image = acquisition_operator.H(
            self.dcf.as_operator()(kdata.data)[0] if self.dcf is not None else kdata.data
        )[0]
        initial_value = torch.cat(
            (initial_image.unsqueeze(0), initial_image.new_zeros((len(regularization_dim), *initial_image.shape))),
            dim=0,
        )

        (tgv_pdhg_output,) = pdhg(
            f=ProximableFunctionalSeparableSum(
                L2NormSquared(target=kdata.data),
                *TotalGeneralizedVariationRegularizedPdhg.get_l1_terms(
                    (self.regularization_weight_grad_term, self.regularization_weight_sym_grad_term)
                ),
            ),
            g=None,
            operator=TgvOperatorMatrix(
                regularization_dim, acquisition_operator, initial_image.data.shape
            ),
            initial_values=(initial_value,),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        img_tensor = TotalGeneralizedVariationRegularizedPdhg.process_pdhg_output(tgv_pdhg_output)
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
