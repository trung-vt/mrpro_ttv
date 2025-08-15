# %% [markdown]
# # Total-generalized-variation (TGV)-minimization reconstruction

# %% [markdown]
# In this notebook, we demonstrate the use of the TGV minimization for image reconstruction.
# We use two examples: denoising and MRI reconstruction with (retrospective) radial undersampling.
# We first use a simple denoising problem on a classical example square image to highlight the strength of TGV.
# Then we retrospectively simulate 2D radial undersampling and apply TGV with `~mrpro.algorithms.optimizers.pdhg` to
# illustrate a real-world application.

# %% [markdown]
# ## Example 1: Denoising

# %% [markdown]
# ### Load data
# Our example data contains a square image used in the paper "Total Generalized Variation":
#
# - ``square.png``
# - ``square_noisy_0_05.png``
#
# We will use TV and TGV regularization for the reconstruction.

# %% mystnb={"code_prompt_show": "Show download details"} tags=["hide-cell"]
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.set_default_device('cuda')

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
# data_folder = Path('data')
data_folder.mkdir(exist_ok=True)
zenodo_get.download(record='16811276', retry_attempts=5, output_dir=data_folder)

# %% tags=["hide-cell"]
import matplotlib.pyplot as plt


def show_images(
    *images: torch.Tensor, titles: list[str] | None = None, clim: tuple[float, float] | None = None
) -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        axes[0][i].imshow(images[i].cpu().squeeze(), cmap='gray', clim=clim)
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


# %% [markdown]
# Load the clean and noisy square images.

# %%
import mrpro
from PIL import Image
from torchvision.transforms.functional import to_tensor

square_clean = to_tensor(Image.open(data_folder / 'square.png').convert('L')).to(device=torch.get_default_device())
square_noisy = to_tensor(Image.open(data_folder / 'square_noisy_0_05.png').convert('L')).to(
    device=torch.get_default_device()
)

show_images(square_noisy, square_clean, titles=['Noisy', 'Clean'], clim=(0, 1))

# %% [markdown]
# Before running the TGV-minimization reconstruction, we first run a TV-regularized reconstruction
# (see <project:tv_minimization_reconstruction.ipynb>).

# %% [markdown]
# ### Set up the operator $A$
# Now, to set up the problem, we need to define the acquisition operator $A$,
# which in this case is just the identity operator.
# We use `~mrpro.operators.IdentityOp` to define the operator.

# %%
identity_op = mrpro.operators.IdentityOp()  # acquisition operator A here is simply the identity operator

# %% [markdown]
# Now we define the routine to run PDHG with TV minimization (see TV notebook for more details).

# %%
from collections.abc import Sequence

from mrpro.operators import (
    FiniteDifferenceOp,
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctionalSeparableSum,
    RearrangeOp,
    ZeroOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared


def tv_minimization_reconstruction(
    measurement: torch.Tensor,
    acquisition_op: LinearOperator,
    grad_term_weight: torch.Tensor,
    dim: Sequence[int] = (-2, -1),
    **pdhg_kwargs,
) -> torch.Tensor:
    """Perform TV-minimization reconstruction."""
    # 1. Compute initial values using the adjoint of the acquisition operator
    adjoint_recon = acquisition_op.adjoint(measurement)[0]
    initial_values = (adjoint_recon,)

    # 2. Define the objective functional
    data_term = 0.5 * L2NormSquared(target=measurement)
    grad_term = L1NormViewAsReal(weight=grad_term_weight)
    minimization_sum = ProximableFunctionalSeparableSum(data_term, grad_term)

    # 3. Define the operator matrix K
    data_term_row = (acquisition_op,)
    nabla = FiniteDifferenceOp(dim=dim, mode='forward')
    grad_term_row = (nabla,)
    operator_matrix = LinearOperatorMatrix((data_term_row, grad_term_row))

    return mrpro.algorithms.optimizers.pdhg(
        f=minimization_sum, g=None, operator=operator_matrix, initial_values=initial_values, **pdhg_kwargs
    )[0]


# %% [markdown]
# Now we can run the PDHG algorithm to solve the TV minimization problem.

# %% tags=["hide-cell"]
# This is a "callback" function that will be called after each iteration of the PDHG algorithm.
# We use it here to print progress information.

from mrpro.algorithms.optimizers.pdhg import PDHGStatus


def callback(optimizer_status: PDHGStatus) -> None:
    """Print the value of the objective functional every 16th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    if iteration % 16 == 0:
        print(f'Iteration {iteration: >3}: Objective = {optimizer_status["objective"](*solution).item():.3e}')


# %%
square_tv_denoised = tv_minimization_reconstruction(
    measurement=square_noisy,
    acquisition_op=identity_op,
    grad_term_weight=torch.tensor(0.05),
    max_iterations=256,
    callback=callback,
)
show_images(square_noisy, square_tv_denoised, square_clean, titles=['Noisy', 'TV Denoised', 'Clean'], clim=(0, 1))


# %% [markdown]
# We can see that TV produces stair-casing effect. We will see that TGV does not suffer from this type of artifacts.

# %% [markdown]
# ## TGV

# %% [markdown]
# Similar to the **scalar TV** notebook, we use $y$ to denote the k-space data of the image $x_{\mathrm{true}}$ sampled
# with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, ...), i.e the forward problem is given as
#
# $$ y = Ax_{\mathrm{true}} + n, $$
#
# where $n$ describes complex Gaussian noise. When using TGV-minimization as regularization method, an approximation of
# $x_{\mathrm{true}}$ is obtained by minimizing the following functional $\mathcal{F}$
#
# $$
# \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2
# + \lambda_1 \| \nabla x - v \|_1
# + \lambda_2 \| \mathcal{E} v \|_1, \quad \quad \quad (1)
# $$
#
# where $\nabla$ is the discretized gradient operator and $\mathcal{E}$ is the discretized
# symmetrized gradient operator.
#
# Similar to the **scalar TV** notebook, we use the
# PDHG-algorithm [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)],
# which is a method for solving problems of the form
#
# $$ \min_x f(K(x)) + g(x)  \quad \quad \quad (2) $$
#
# where $f$ and $g$ denote proper, convex, lower-semicontinous functionals and $K$ denotes a linear operator.

# %% [markdown]
# ### Recast the problem to be able to apply PDHG
# To apply the PDHG algorithm for TGV, we need to recast the problem into the form of (2). We need to identify
# the functionals $f$ and $g$ and the operator $K$. We chose an identification for which both
# $\mathrm{prox}_{\sigma f^{\ast}}$ and $\mathrm{prox}_{\tau g}$ are easy to compute:
#
# #### $f(z) = f(p,q,r) = f_1(p) + f_2(q) + f_3(r) = \frac{1}{2}\|p - y\|_2^2 + \lambda_1 \|q\|_1 + \lambda_0 \|r\|_1.$

# %% [markdown]
# #### $K(x) = [[A, 0], [\nabla, -1], [0, \mathcal{E}]]^T$
#
#   where $\nabla$ is the finite difference operator that computes the directional derivatives along the last two
#   dimensions (y,x), implemented as `~mrpro.operators.FiniteDifferenceOp`, and
#  `~mrpro.operators.LinearOperatorMatrix` can be used to stack the operators.

# %% [markdown]
# #### $g(x) = 0,$
#
# implemented as `~mrpro.operators.functionals.ZeroFunctional`

# %% [markdown]
# This identification allows us to compute the proximal operators of $f$ and $g$ easily.


# %%
def tgv_minimization_reconstruction(
    measurement: torch.Tensor,
    acquisition_op: LinearOperator,
    grad_term_weight: torch.Tensor,
    sym_grad_term_weight: torch.Tensor,
    dim: Sequence[int] = (-2, -1),
    **pdhg_kwargs,
) -> torch.Tensor:
    """Perform TGV-minimization reconstruction."""
    # 1. Compute initial values using the adjoint of the acquisition operator
    adjoint_recon = acquisition_op.adjoint(measurement)[0]
    # Auxiliary tensor v which is in the gradient domain.
    auxiliary_v_tensor = adjoint_recon.new_zeros((len(dim), *adjoint_recon.shape))
    # Increase the number of dimensions of initial image by one.
    # Must make the number of dimensions of the elements in initial values list match one another,
    # because (currently) pdhg function concatenates the individual norms of each operator
    # in a matrix's row and the norm has the same shape as the input.
    # If the number of dimensions don't match, we get a runtime error like this:
    #   RuntimeError: stack expects each tensor to be equal size, but got ... at entry 0 and ... at entry 1
    adjoint_recon = adjoint_recon.unsqueeze(0)
    initial_values = (adjoint_recon, auxiliary_v_tensor)

    # 2. Define the objective functional
    data_term = 0.5 * L2NormSquared(target=measurement)
    grad_term = L1NormViewAsReal(weight=grad_term_weight)
    sym_grad_term = L1NormViewAsReal(weight=sym_grad_term_weight)
    minimization_sum = ProximableFunctionalSeparableSum(data_term, grad_term, sym_grad_term)

    # 3. Define the operator matrix K
    # 3.1. First row (corresponding to the L2-norm data term): Ax + 0
    data_term_row = (acquisition_op, ZeroOp())

    # 3.2. Second row (corresponding to the first L1-norm regularization term): \nabla x - v
    # Reduce the number of dimensions of the image by one before applying the finite difference operator
    # to make the output of the finite difference operator match the auxiliary tensor v.
    squeeze_op = RearrangeOp('1 ... -> ...')
    forward_nabla = FiniteDifferenceOp(dim=dim, mode='forward')
    grad_term_row = (forward_nabla @ squeeze_op, -1 * IdentityOp())

    # 3.3. Third row (corresponding to the second L1-norm regularization term): 0 + \mathcal{E} v
    backward_nabla = FiniteDifferenceOp(dim=dim, mode='backward')
    transpose_op = RearrangeOp('sym_grad_dim  grad_dim  ...   ->   grad_dim  sym_grad_dim  ...')
    symmetric_gradient_op = 0.5 * (1 + transpose_op) @ backward_nabla
    sym_grad_term_row = (ZeroOp(), symmetric_gradient_op)

    operator_matrix = LinearOperatorMatrix((data_term_row, grad_term_row, sym_grad_term_row))

    return mrpro.algorithms.optimizers.pdhg(
        f=minimization_sum,
        g=None,  # automatically converted to zero functionals
        operator=operator_matrix,
        initial_values=initial_values,
        **pdhg_kwargs,
    )[0]


# %% [markdown]
# Show the TGV result.

# %%
square_tgv_denoised = tgv_minimization_reconstruction(
    measurement=square_noisy,
    acquisition_op=identity_op,
    grad_term_weight=torch.tensor(0.05),
    sym_grad_term_weight=torch.tensor(0.1),
    max_iterations=256,
    callback=callback,
)
show_images(
    square_noisy,
    square_tv_denoised,
    square_tgv_denoised,
    square_clean,
    titles=['Noisy', 'TV Denoised', 'TGV Denoised', 'Clean'],
    clim=(0, 1),
)

# %% [markdown]
# We can see that TGV does not contain stair-casing artifacts like TV.

# %% [markdown]
# ### Second example: Radial undersampling

# %% [markdown]
# Download the 4-coil cartesian k-space data of a brain MRI scan provided by the author of TGV method
# (reference: [bredies paper](url))

# %%
# Download ground-truth k-space data from Zenodo
zenodo_get.download(record='800525', retry_attempts=5, output_dir=data_folder)

# %% [markdown]
# Load brain k-space data and perform adjoint to obtain the ground truth.

# %%
from einops import rearrange
from mrpro.data import SpatialDimension
from scipy.io import loadmat

file_name = '1_rawdata_brainT2_4ch.mat'
kdata_true = torch.tensor(loadmat(data_folder / file_name)['rawdata'])
kdata_true = rearrange(kdata_true, 'k1 k0 coils  ->  1 coils 1 k1 k0')
# kdata = torch.flip(kdata, dims=(-2, -1))
kdata_true = kdata_true.to(dtype=torch.complex64)  # If default dtype is torch.float32, use complex64

recon_matrix = SpatialDimension(z=kdata_true.shape[-3], y=kdata_true.shape[-2], x=kdata_true.shape[-1])
encoding_matrix = SpatialDimension(z=kdata_true.shape[-3], y=kdata_true.shape[-2], x=kdata_true.shape[-1])

fourier_op = mrpro.operators.FastFourierOp(
    dim=(-2, -1),
    recon_matrix=recon_matrix,
    encoding_matrix=encoding_matrix,
)
x_true = fourier_op(kdata_true)[0]

x_true_rss = x_true.abs().square().sum(dim=-4).sqrt().squeeze()
show_images(x_true_rss, titles=['Ground Truth'], clim=(0, 7e-4))

# %% [markdown]
# ### Set up the operator $A$

# %% [markdown]
# We first define the undersampling operator by generating a Cartesian mask with 48 radial spokes
# and use it to create an `~mrpro.operators.CartesianMaskingOp` operator object.

# %%
from torchvision.transforms.functional import rotate


def radial_mask(ny: int, nx: int, num_spokes: int) -> torch.Tensor:
    """Generate a radial mask with the specified number of spokes."""
    theta = 180 * (3.0 - 5**0.5)  # golden angle ~137.508°
    mask = torch.zeros((1, 1, ny, nx), dtype=torch.bool)

    # prototype spoke: horizontal line through center
    base_spoke = torch.zeros((1, 1, ny, nx))
    base_spoke[0, 0, ny // 2, :] = 1.0

    for i_spoke in range(num_spokes):
        spoke = rotate(base_spoke, angle=theta * i_spoke, fill=0)
        mask |= spoke > 0.5
    return mask


Nx, Ny = 256, 256
num_spokes = 48
mask_op = mrpro.operators.CartesianMaskingOp(radial_mask(Ny, Nx, num_spokes))
show_images(torch.tensor(mask_op.mask), titles=[f'Mask ({num_spokes} spokes)'])

# %% [markdown]
# Then we define the acquisition operator $A$ by combining the masking operator with the FFT operator.

# %%

acquisition_op = mask_op @ fourier_op
# Apply A(x_true) to get undersampled k-space data
kdata_undersampled = acquisition_op(x_true)[0]
x_adjoint_recon = acquisition_op.adjoint(kdata_undersampled)[0]
x_adjoint_recon_rss = torch.sum(x_adjoint_recon.abs() ** 2, dim=1).sqrt()
show_images(x_adjoint_recon_rss, x_true_rss, titles=['Adjoint', 'Ground Truth'], clim=(0, 7e-4))

# %%
x_tv_recon = tv_minimization_reconstruction(
    measurement=kdata_undersampled,
    acquisition_op=acquisition_op,
    grad_term_weight=torch.tensor(5e-6),
    max_iterations=257,
    callback=callback,
)
x_tv_recon_rss = torch.sum(x_tv_recon.abs() ** 2, dim=-4).sqrt()
show_images(x_adjoint_recon_rss, x_tv_recon_rss, x_true_rss, titles=['Adjoint', 'TV', 'Ground Truth'], clim=(0, 7e-4))

# %%
x_tgv_recon = tgv_minimization_reconstruction(
    measurement=kdata_undersampled,
    acquisition_op=acquisition_op,
    grad_term_weight=torch.tensor(5e-6),
    sym_grad_term_weight=torch.tensor(10e-6),
    max_iterations=257,
    callback=callback,
)
x_tgv_recon_rss = torch.sum(x_tgv_recon.abs() ** 2, dim=-4).sqrt()
show_images(
    x_adjoint_recon_rss,
    x_tv_recon_rss,
    x_tgv_recon_rss,
    x_true_rss,
    titles=['Adjoint', 'TV', 'TGV', 'Ground Truth'],
    clim=(0, 7e-4),
)

# %% [markdown]
# ### Compare the results
# We now compare the close-up results of the TV-minimization and the TGV-minimization.
# Once again, we can see the stair-casing artifacts on the TV reconstruction which is effectively absent on TGV.

# %%
show_images(
    x_adjoint_recon_rss[..., :128, :128],
    x_tv_recon_rss[..., :128, :128],
    x_tgv_recon_rss[..., :128, :128],
    x_true_rss[..., :128, :128],
    titles=['Adjoint', 'TV', 'TGV', 'Ground Truth'],
    clim=(0, 7e-4),
)


# %%
show_images(
    x_tgv_recon[0, 0, 0, 0, :128, :128].abs().squeeze(),
    x_tgv_recon[0, 0, 1, 0, :128, :128].abs().squeeze(),
    x_tgv_recon[0, 0, 2, 0, :128, :128].abs().squeeze(),
    x_tgv_recon[0, 0, 3, 0, :128, :128].abs().squeeze(),
    titles=['1', '2', '3', '4'],
    clim=(0, 7e-4),
)


# %% [markdown]
# We have successfully performed denoising and reconstruction using TGV-minimization.
#
# ### Next steps
# Play around with the regularization weights and the number of iterations to see how they affect the final image.
# You can also vary the amount of noise or the number of spokes to see how the reconstruction quality changes.
