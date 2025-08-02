"""Variational-Regularized Reconstruction using PDHG."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from mrpro.algorithms import VariationalRegularizedPdhg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators.LinearOperator import LinearOperator


class VariationalRegularizedReconstruction(DirectReconstruction):
    """Variational-regularized reconstruction (TV, TGV, etc.)."""

    variational_regularized_pdhg: VariationalRegularizedPdhg
    """Variational regularized PDHG algorithm (TV, TGV, etc.)."""

    regularization_weight: Any | None = None

    def __init__(
        self,
        variational_regularized_pdhg: VariationalRegularizedPdhg,
        regularization_weight: Any | None = None,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        variational_regularized_pdhg
            Variational regularized PDHG algorithm (TV, TGV, etc.).
        regularization_weight
            Regularization weight for the variational regularized PDHG algorithm.
            Typically reset every time PDHG is run.
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

        Raises
        ------
        ValueError
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is `None`.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.variational_regularized_pdhg = variational_regularized_pdhg
        self.regularization_weight = regularization_weight

    def set_regularization_weight(self, regularization_weight: Any) -> None:
        """Set the regularization weight for the variational regularized PDHG algorithm.

        Parameters
        ----------
        regularization_weight
            Regularization weight for the variational regularized PDHG algorithm.
        """
        self.regularization_weight = regularization_weight

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            K-space data to reconstruct.
        **kwargs
            Additional keyword arguments for the variational regularized PDHG algorithm.

        Returns
        -------
            The reconstructed image.
        """
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)

        acquisition_operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op

        initial_image = acquisition_operator.H(
            self.dcf.as_operator()(kdata.data)[0] if self.dcf is not None else kdata.data
        )[0]

        img_tensor: torch.Tensor = self.variational_regularized_pdhg(
            acquisition_operator=acquisition_operator,
            regularization_weight=self.regularization_weight,
            measurement=kdata.data,
            initial_image=initial_image,
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
