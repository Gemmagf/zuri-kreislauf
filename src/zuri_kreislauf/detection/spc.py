"""Classical multivariate statistical process control (Hotelling T2 + Q/SPE).

Chosen over deep-learning anomaly detectors because the monitored dataset has
n=78 monthly observations — nowhere near enough to fit, let alone validate, a
neural detector. T2/Q with a robust (MinCovDet) covariance estimate is the
standard SPC tool for exactly this regime: few observations, a handful of
correlated process variables, and a need for statistically grounded control
limits rather than an arbitrary threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class SPCResult:
    t2: pd.Series
    t2_ucl: float
    q: pd.Series
    q_ucl: float
    t2_contributions: pd.DataFrame  # per-feature contribution to each month's T2
    q_contributions: pd.DataFrame  # per-feature contribution to each month's Q

    @property
    def flags(self) -> pd.Series:
        """Months where either statistic exceeds its control limit."""
        return (self.t2 > self.t2_ucl) | (self.q > self.q_ucl)


def _hotelling_t2_ucl(n: int, p: int, alpha: float = 0.01) -> float:
    """Phase-I UCL for T2 under a Gaussian process assumption (Tracy et al. 1992)."""
    f_crit = stats.f.ppf(1 - alpha, p, n - p)
    return p * (n - 1) * (n + 1) / (n * (n - p)) * f_crit


def run_spc(
    ratios: pd.DataFrame,
    columns: tuple[str, ...],
    alpha: float = 0.01,
    n_components: int | None = None,
) -> SPCResult:
    """Fit a robust-covariance T2 monitor and a PCA-residual Q monitor.

    Both are fit on the full `ratios` history (Phase-I / retrospective
    monitoring — appropriate here since we're auditing history, not
    screening new incoming months against a fixed baseline).
    """
    x = ratios[list(columns)].dropna()
    n, p = x.shape
    if n_components is None:
        n_components = max(1, p // 2)

    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    # --- Hotelling T2 on a robust covariance estimate ---
    mcd = MinCovDet(random_state=0).fit(z)
    cov_inv = np.linalg.pinv(mcd.covariance_)
    centered = z - mcd.location_
    t2_values = np.einsum("ij,jk,ik->i", centered, cov_inv, centered)
    t2 = pd.Series(t2_values, index=x.index, name="t2")
    t2_ucl = _hotelling_t2_ucl(n, p, alpha)

    # Per-feature T2 contribution (Kourti & MacGregor 1996 partial-contribution
    # style): contribution_ij = z_ij * (Sigma^-1 @ z_i)_j, which sums exactly
    # to T2_i across features j.
    t2_contrib = pd.DataFrame(centered * (centered @ cov_inv), index=x.index, columns=x.columns)

    # --- Q / SPE on PCA reconstruction residual ---
    pca = PCA(n_components=n_components, random_state=0).fit(z)
    z_hat = pca.inverse_transform(pca.transform(z))
    residual = z - z_hat
    q_values = np.sum(residual**2, axis=1)
    q = pd.Series(q_values, index=x.index, name="q")

    # Q control limit via Jackson & Mudholkar (1979) chi-square approximation
    # fit to the empirical residual eigenvalue spectrum from the *unused*
    # components.
    explained = pca.explained_variance_
    total_var = np.var(z, axis=0, ddof=1).sum()
    unused_var = max(total_var - explained.sum(), 1e-9)
    residual_eigs = np.full(p - n_components, unused_var / max(p - n_components, 1))
    theta1, theta2, theta3 = (np.sum(residual_eigs**k) for k in (1, 2, 3))
    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2) if theta2 > 0 else 1.0
    z_alpha = stats.norm.ppf(1 - alpha)
    q_ucl = theta1 * (
        (z_alpha * np.sqrt(2 * theta2 * h0**2) / theta1 + 1 + theta2 * h0 * (h0 - 1) / theta1**2)
        ** (1 / h0)
    )

    q_contrib = pd.DataFrame(residual**2, index=x.index, columns=x.columns)

    return SPCResult(
        t2=t2,
        t2_ucl=t2_ucl,
        q=q,
        q_ucl=q_ucl,
        t2_contributions=t2_contrib,
        q_contributions=q_contrib,
    )


def top_contributors(contributions: pd.DataFrame, month: pd.Timestamp, k: int = 3) -> pd.Series:
    """The k features contributing most to a flagged month's statistic."""
    return contributions.loc[month].sort_values(ascending=False).head(k)
