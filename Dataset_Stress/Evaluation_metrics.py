# %%
# Evaluate_dat.py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from typing import Tuple

def compute_dimensionwise_probability(real: np.ndarray, synth: np.ndarray, bandwidth: float = None) -> np.ndarray:
    """
    For each feature/dimension, estimate a kernel density from `real` and return
    the average probability density of `synth` samples under that KDE.
    Args:
        real: array shape (n_real, n_features)
        synth: array shape (n_synth, n_features)
        bandwidth: optional KDE bandwidth; if None, use scipy default
    Returns:
        probs: array shape (n_features,) with mean density of synth under real KDE per feature
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("real and synth must be 2D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synth must have same number of features")
    n_features = real.shape[1]
    probs = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        # 1D KDE per feature
        data = real[:, j]
        kde = gaussian_kde(data, bw_method=bandwidth)
        densities = kde.evaluate(synth[:, j])
        probs[j] = float(np.mean(densities))
    return probs

def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = "rbf", gamma: float = None) -> float:
    """
    Compute unbiased estimate of squared Maximum Mean Discrepancy (MMD^2)
    between samples X and Y using an RBF (Gaussian) kernel or linear kernel.
    Args:
        X: array shape (n_x, d)
        Y: array shape (n_y, d)
        kernel: 'rbf' or 'linear'
        gamma: kernel bandwidth parameter for rbf; if None, use median heuristic
    Returns:
        mmd2: float, estimated MMD^2
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same dimensionality")
    def rbf_kernel(A, B, gamma):
        D2 = cdist(A, B, 'sqeuclidean')
        return np.exp(-gamma * D2)
    def linear_kernel(A, B):
        return A.dot(B.T)

    if kernel == "linear":
        Kxx = linear_kernel(X, X)
        Kyy = linear_kernel(Y, Y)
        Kxy = linear_kernel(X, Y)
    elif kernel == "rbf":
        # median heuristic for gamma if not provided
        if gamma is None:
            Z = np.vstack([X, Y])
            dists = cdist(Z, Z, 'sqeuclidean')
            med = np.median(dists[dists > 0])
            gamma = 1.0 / (med + 1e-12)
        Kxx = rbf_kernel(X, X, gamma)
        Kyy = rbf_kernel(Y, Y, gamma)
        Kxy = rbf_kernel(X, Y, gamma)
    else:
        raise ValueError("Unsupported kernel: choose 'rbf' or 'linear'")

    nx = X.shape[0]
    ny = Y.shape[0]
    # unbiased estimator
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (nx * (nx - 1))
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (ny * (ny - 1))
    sum_xy = np.sum(Kxy) / (nx * ny)
    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return float(mmd2)


# %%
# Evaluate_fairness.py
import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.metrics import davies_bouldin_score as sk_davies_bouldin_score
from typing import Sequence

def compute_cluster_statistical_parity(
    cluster_labels: np.ndarray,
    protected: np.ndarray
) -> float:
    """
    Unsupervised statistical parity over clusters.
    Measures average absolute difference of cluster assignment rates
    across protected groups.
    """
    cluster_labels = np.asarray(cluster_labels)
    protected = np.asarray(protected)

    if len(cluster_labels) != len(protected):
        raise ValueError("cluster_labels and protected must have same length")

    if not ({0, 1} <= set(np.unique(protected))):
        raise ValueError("protected must be binary")

    disparities = []
    clusters = np.unique(cluster_labels)

    for c in clusters:
        p1 = np.mean(cluster_labels[protected == 1] == c)
        p0 = np.mean(cluster_labels[protected == 0] == c)
        disparities.append(abs(p1 - p0))

    return float(np.mean(disparities))


def silhouette_score(X: np.ndarray, labels: Sequence[int], metric: str = "euclidean") -> float:
    """
    Wrapper around sklearn.metrics.silhouette_score.
    Args:
        X: feature matrix (n_samples, n_features)
        labels: cluster labels for each sample
        metric: distance metric
    Returns:
        silhouette score (float)
    """
    return float(sk_silhouette_score(X, labels, metric=metric))

def davies_bouldin_score(X: np.ndarray, labels: Sequence[int]) -> float:
    """
    Wrapper around sklearn.metrics.davies_bouldin_score.
    Args:
        X: feature matrix (n_samples, n_features)
        labels: cluster labels
    Returns:
        Davies-Bouldin index (float) lower is better
    """
    return float(sk_davies_bouldin_score(X, labels))


# %%
# Privact_eval.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from typing import Tuple

def compute_epsilon_identifiability(real: np.ndarray, synth: np.ndarray, bandwidth: float = None) -> float:
    """
    Heuristic estimate of an identifiability epsilon using density ratio:
    epsilon = max_x log( p_real(x) / p_synth(x) )
    We estimate p_real and p_synth with 1D-per-feature product of KDEs (independence assumption)
    Args:
        real: (n_real, d)
        synth: (n_synth, d)
        bandwidth: KDE bandwidth passed to gaussian_kde
    Returns:
        epsilon (float): max log density ratio across real samples
    Notes:
        This is a heuristic, not a formal DP epsilon.
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("real and synth must be 2D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synth must have same number of features")
    d = real.shape[1]
    # build per-dimension KDEs
    kde_real = [gaussian_kde(real[:, j], bw_method=bandwidth) for j in range(d)]
    kde_synth = [gaussian_kde(synth[:, j], bw_method=bandwidth) for j in range(d)]
    # evaluate log density ratio for each real sample
    log_ratios = []
    for i in range(real.shape[0]):
        x = real[i]
        logp_real = 0.0
        logp_synth = 0.0
        for j in range(d):
            pr = max(kde_real[j].evaluate(x[j])[0], 1e-300)
            ps = max(kde_synth[j].evaluate(x[j])[0], 1e-300)
            logp_real += np.log(pr)
            logp_synth += np.log(ps)
        log_ratios.append(logp_real - logp_synth)
    epsilon = float(np.max(log_ratios))
    return epsilon

def compute_nndr(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """
    Compute Nearest Neighbor Distance Ratio (NNDR) for each point in `original`:
    ratio = dist_to_nearest_in_candidate / dist_to_nearest_in_original_excluding_self
    Args:
        original: (n_orig, d)
        candidate: (n_cand, d) typically synthetic
    Returns:
        ratios: array shape (n_orig,) of NNDR values
    """
    original = np.asarray(original)
    candidate = np.asarray(candidate)
    if original.ndim != 2 or candidate.ndim != 2:
        raise ValueError("original and candidate must be 2D arrays")
    # nearest neighbor in candidate
    nn_cand = NearestNeighbors(n_neighbors=1).fit(candidate)
    d_cand, _ = nn_cand.kneighbors(original, return_distance=True)
    d_cand = d_cand.ravel()
    # nearest neighbor in original excluding self: use n_neighbors=2 and take second neighbor
    nn_orig = NearestNeighbors(n_neighbors=2).fit(original)
    d_orig_all, idx = nn_orig.kneighbors(original, return_distance=True)
    # d_orig_all[:,0] is zero (self), so take second column
    if d_orig_all.shape[1] < 2:
        # fallback: use single neighbor distances (may be zero)
        d_orig = d_orig_all[:, 0]
    else:
        d_orig = d_orig_all[:, 1]
    # avoid division by zero
    d_orig = np.maximum(d_orig, 1e-12)
    ratios = d_cand / d_orig
    return ratios


# %%
# Utility_eval.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

def compute_beta_recall(original: np.ndarray, synth: np.ndarray, beta: float) -> float:
    """
    Beta-recall: fraction of original records that have at least one synthetic neighbor
    within radius `beta`.
    Args:
        original: (n_orig, d)
        synth: (n_synth, d)
        beta: radius threshold
    Returns:
        recall (float) in [0,1]
    """
    original = np.asarray(original)
    synth = np.asarray(synth)
    if original.ndim != 2 or synth.ndim != 2:
        raise ValueError("original and synth must be 2D arrays")
    nbrs = NearestNeighbors(radius=beta).fit(synth)
    # radius_neighbors returns list of neighbors per sample
    neigh_indices = nbrs.radius_neighbors(original, return_distance=False)
    has_neighbor = np.array([len(nei) > 0 for nei in neigh_indices])
    return float(np.mean(has_neighbor))

def compute_alpha_precision(original: np.ndarray, synth: np.ndarray, alpha: float) -> float:
    """
    Alpha-precision: fraction of synthetic records that are within radius `alpha`
    of at least one original record.
    Args:
        original: (n_orig, d)
        synth: (n_synth, d)
        alpha: radius threshold
    Returns:
        precision (float) in [0,1]
    """
    original = np.asarray(original)
    synth = np.asarray(synth)
    if original.ndim != 2 or synth.ndim != 2:
        raise ValueError("original and synth must be 2D arrays")
    nbrs = NearestNeighbors(radius=alpha).fit(original)
    neigh_indices = nbrs.radius_neighbors(synth, return_distance=False)
    has_neighbor = np.array([len(nei) > 0 for nei in neigh_indices])
    return float(np.mean(has_neighbor))


# %%
# Add to Evaluate_dat.py
import numpy as np
from typing import Union, Optional
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    *,
    discrete_target: Optional[bool] = None,
    n_neighbors: int = 3,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    return_mean: bool = False
) -> Union[np.ndarray, float]:
    """
    Compute mutual information between each column of X and target y.

    Uses scikit-learn's mutual_info_regression for continuous targets and
    mutual_info_classif for discrete targets. If `discrete_target` is None,
    the function will attempt to infer target type (integer-like -> discrete).

    Args:
        X: array of shape (n_samples, n_features).
        y: array of shape (n_samples,).
        discrete_target: if True, treat y as discrete (classification);
                         if False, treat y as continuous (regression);
                         if None, infer from y dtype/values.
        n_neighbors: number of neighbors used by the underlying estimator.
        random_state: random state passed to the estimator for reproducibility.
        return_mean: if True, return the mean mutual information across features;
                     otherwise return an array of MI values per feature.

    Returns:
        If return_mean is False: np.ndarray of shape (n_features,) with MI values.
        If return_mean is True: float, mean MI across features.

    Raises:
        ValueError: if X and y have incompatible shapes.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features)")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array (n_samples,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    # Infer target type if not provided
    if discrete_target is None:
        # treat integer-like targets with few unique values as discrete
        if np.issubdtype(y.dtype, np.integer):
            discrete_target = True
        else:
            # if number of unique values is small relative to samples, treat as discrete
            unique_ratio = float(np.unique(y).size) / float(y.size)
            discrete_target = unique_ratio < 0.05

    if discrete_target:
        mi = mutual_info_classif(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state)
    else:
        mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state)

    mi = np.asarray(mi, dtype=float)
    return float(mi.mean()) if return_mean else mi


# %%

# Add to Evaluate_dat.py
import numpy as np
from typing import Union, Optional, Tuple
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.utils import check_random_state

def compute_ks(
    real: np.ndarray,
    synth: np.ndarray,
    *,
    return_pvalue: bool = False,
    axis: int = 1
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Kolmogorov-Smirnov two-sample statistic for each feature.

    Args:
        real: array shape (n_real, n_features)
        synth: array shape (n_synth, n_features)
        return_pvalue: if True, also return p-values per feature
        axis: axis representing features (default 1 for shape (n_samples, n_features))

    Returns:
        If return_pvalue is False: ks_stats: np.ndarray shape (n_features,)
        If return_pvalue is True: (ks_stats, p_values) each shape (n_features,)
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("real and synth must be 2D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synth must have the same number of features")
    n_features = real.shape[1]
    ks_stats = np.zeros(n_features, dtype=float)
    p_values = np.zeros(n_features, dtype=float) if return_pvalue else None
    for j in range(n_features):
        stat, p = ks_2samp(real[:, j], synth[:, j])
        ks_stats[j] = float(stat)
        if return_pvalue:
            p_values[j] = float(p)
    return (ks_stats, p_values) if return_pvalue else ks_stats

def compute_wasserstein(
    real: np.ndarray,
    synth: np.ndarray,
    *,
    return_per_feature: bool = True
) -> Union[np.ndarray, float]:
    """
    Compute 1D Wasserstein (Earth Mover's) distance for each feature.

    Args:
        real: array shape (n_real, n_features)
        synth: array shape (n_synth, n_features)
        return_per_feature: if True return array of distances per feature,
                            otherwise return mean distance across features.

    Returns:
        np.ndarray of shape (n_features,) if return_per_feature True,
        else float (mean distance).
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("real and synth must be 2D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synth must have the same number of features")
    n_features = real.shape[1]
    dists = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        dists[j] = float(wasserstein_distance(real[:, j], synth[:, j]))
    return dists if return_per_feature else float(dists.mean())

def compute_dwp(
    real: np.ndarray,
    synth: np.ndarray,
    *,
    n_permutations: int = 1000,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    return_per_feature: bool = True
) -> Union[np.ndarray, float]:
    """
    Compute permutation-based p-values for the observed 1D Wasserstein distance
    between `real` and `synth` for each feature. This performs a two-sample
    permutation test where the test statistic is the Wasserstein distance.

    Args:
        real: array shape (n_real, n_features)
        synth: array shape (n_synth, n_features)
        n_permutations: number of permutations to estimate p-values (default 1000)
        random_state: seed or RandomState for reproducibility
        return_per_feature: if True return array of p-values per feature,
                            otherwise return mean p-value across features.

    Returns:
        np.ndarray of p-values shape (n_features,) if return_per_feature True,
        else float (mean p-value).
    Notes:
        Permutation tests can be slow for large n_permutations and many features.
    """
    rng = check_random_state(random_state)
    real = np.asarray(real)
    synth = np.asarray(synth)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("real and synth must be 2D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synth must have the same number of features")
    n_features = real.shape[1]
    p_values = np.zeros(n_features, dtype=float)
    combined = np.vstack([real, synth])
    n_real = real.shape[0]
    n_total = combined.shape[0]

    # Precompute indices for permutations to avoid repeated allocations
    for j in range(n_features):
        obs = float(wasserstein_distance(real[:, j], synth[:, j]))
        perm_stats = np.zeros(n_permutations, dtype=float)
        for k in range(n_permutations):
            perm_idx = rng.permutation(n_total)
            perm_real_idx = perm_idx[:n_real]
            perm_synth_idx = perm_idx[n_real:]
            perm_real = combined[perm_real_idx, j]
            perm_synth = combined[perm_synth_idx, j]
            perm_stats[k] = wasserstein_distance(perm_real, perm_synth)
        # p-value: proportion of permuted stats >= observed (one-sided)
        p = (np.sum(perm_stats >= obs) + 1) / (n_permutations + 1)
        p_values[j] = float(p)

    return p_values if return_per_feature else float(p_values.mean())


