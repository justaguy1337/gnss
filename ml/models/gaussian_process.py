"""
Gaussian Process Residual Model (Paper §III-G)
===============================================
GP with Matérn(ν=2.5) + Periodic kernel for residual correction
and uncertainty quantification.

Paper §III-G (eqs. 16-21):
  r_h = y_h - ŷ_h
  k(x,x') = k_Matérn(x,x') + k_Periodic(x,x')
  ŷ_final = ŷ_h + μ_GP
  uncertainty = σ_GP

Fix from original ridge_stacker_gnss.py:
  - Matérn ν=2.5 (was 1.5) for smoother residuals per paper
"""

import numpy as np
import torch
import gpytorch
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GP_MATERN_NU, GP_TRAIN_ITERS, GP_LEARNING_RATE


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP with composite kernel for GNSS residual modeling.

    Paper §III-G:
      "the GP uses a mix of kernels — specifically k_Matérn joined
       with k_Periodic — to handle different patterns"
      "one piece models rough but continuous leftovers, another
       picks up repeating signals tied to orbit timing"
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Composite kernel (paper eq. 17):
        # k(x,x') = k_Matérn(x,x') + k_Periodic(x,x')
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=GP_MATERN_NU)
            + gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def train_gp(
    X_train: np.ndarray,
    residuals: np.ndarray,
    n_iter: int = GP_TRAIN_ITERS,
    verbose: bool = False
) -> Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """
    Train Gaussian Process on stacker residuals.

    Paper §III-G:
      "values for these parts adjust by maximizing the log marginal
       likelihood of observed data" (eq. 18)

    Parameters
    ----------
    X_train : (N,) — time indices (normalized)
    residuals : (N,) — actual - stacker prediction (r_h = y_h - ŷ_h)
    n_iter : int — optimization iterations

    Returns
    -------
    gp : ExactGPModel (in eval mode)
    likelihood : GaussianLikelihood (in eval mode)
    """
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(residuals, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = ExactGPModel(train_x, train_y, likelihood)

    # Training mode
    gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=GP_LEARNING_RATE)

    # Maximize log marginal likelihood (paper eq. 18)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = gp(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if verbose and (i + 1) % 25 == 0:
            print(f"    GP iter {i+1}/{n_iter} | loss={loss.item():.4f}")

    # Switch to eval mode
    gp.eval()
    likelihood.eval()

    return gp, likelihood


def predict_gp(
    gp: ExactGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GP posterior prediction.

    Paper §III-G (eqs. 19-20):
      μ_GP(x*) = K*ᵀ K⁻¹ y
      σ²_GP(x*) = K(x*,x*) - K*ᵀ K⁻¹ K*

    Returns
    -------
    mean : np.ndarray — GP mean correction (μ_GP)
    std : np.ndarray — GP uncertainty (σ_GP)
    """
    test_x = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(gp(test_x))

    return pred.mean.numpy(), pred.stddev.numpy()


def apply_gp_correction(
    stacker_pred: np.ndarray,
    gp_mean: np.ndarray
) -> np.ndarray:
    """
    Final prediction with GP correction (paper eq. 21):
      ŷ_final = ŷ_h + μ_GP
    """
    return stacker_pred + gp_mean
