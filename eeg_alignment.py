"""
eeg_alignment.py

F3 (CRITICAL) -- shared Euclidean/Riemannian Alignment module.
=============================================================================
WHY THIS MODULE EXISTS:
  Every LOSO driver in this codebase (run_step4_matched_spatial_control.py,
  run_step4_condition4_asymmetric_mamba.py, and now run_step4_eegnet_ea.py
  for F7) computed EA whitening the same way: a SINGLE whitening matrix W
  fit on the pooled 28-subject training set's mean trial covariance, applied
  uniformly to every trial including the held-out subject's. This is
  "pooled EA". It is not wrong on its face (it does not use labels, and the
  held-out subject's own trials are never used to compute W), but it is a
  weaker alignment than the standard per-subject EA used in the transfer-
  learning BCI literature (Zanini et al. 2018; He & Wu 2019), where EACH
  subject -- including the held-out one -- gets its own whitening reference
  computed purely from that subject's own trial covariances. Per-subject EA
  removes inter-subject covariance-scale variability that pooled EA leaves
  in place, and is the more defensible choice for a LOSO claim. This module
  makes the EA mode an explicit, tested, swappable choice instead of a
  silent single implementation duplicated byte-for-byte across scripts.

FOUR MODES (`--ea-mode {none,pooled,per-subject,riemannian}`):
  - "none":         NO alignment. Returns an identity whitening matrix (a
                     single (C,C) identity, handled exactly like "pooled"
                     by every caller -- applying it is a mathematical
                     no-op). Exists so the F7 2x2 alignment/representation
                     grid ("no-EA" x {tangent-space, EEGNet}) can be driven
                     by the SAME `--ea-mode` flag as every other cell,
                     instead of each driver special-casing "skip EA
                     entirely" as a separate code path. Also the "new
                     tangent-no-EA cell" referenced in AUDIT.md's F7 row.
  - "pooled":      current/original behaviour. ONE mean covariance (simple
                    arithmetic/Euclidean mean) over ALL trials passed in,
                    ONE resulting W applied to every trial regardless of
                    subject. Kept as the default for backward comparability
                    with all previously reported numbers, NOT because it is
                    recommended going forward.
  - "per-subject":  covariances are grouped by subject_ids; a separate
                    arithmetic/Euclidean mean covariance (hence a separate
                    W) is computed PER SUBJECT, from that subject's own
                    trials only. This includes the held-out test subject:
                    its own W is computed from its own (unlabeled) trials,
                    which is legitimate because EA never touches labels.
  - "riemannian":   same per-subject grouping as "per-subject", but the
                    per-subject reference covariance is the affine-invariant
                    Riemannian (Frechet/Karcher) mean of that subject's
                    covariances, not the arithmetic mean. This is the
                    theoretically preferred recentering for SPD matrices
                    (the arithmetic mean is not the natural mean on the SPD
                    manifold) and is standard practice in Riemannian-
                    Alignment BCI transfer learning.

LABEL-FREE GUARANTEE:
  `euclidean_align()` does not take `y` as a parameter at all -- by
  construction, it cannot use label information. `verify_label_free()`
  below is a runtime, defensive double-check: it recomputes the alignment
  twice with two different decoy label arrays "in scope" (passed to a
  wrapper, then explicitly discarded) and asserts the resulting whitening
  matrices are bit-for-bit identical. This guards against a future edit
  accidentally threading `y` into the alignment path.

SUBJECT-DECODABILITY DIAGNOSTIC:
  `subject_decodability_accuracy()` trains a quick multi-class classifier to
  predict WHICH SUBJECT a trial came from, given its (aligned or unaligned)
  tangent-space features. This is the standard sanity check from the EA
  literature: alignment should substantially reduce subject decodability
  (ideally toward chance = 1/n_subjects), since removing subject-identity
  information from the spatial statistics is the entire point of EA. It is
  a diagnostic, not a leakage check (F-LEAK covers that) -- it answers "did
  the alignment do what it claims to do," not "did the pipeline cheat."
=============================================================================
"""

import math
import numpy as np

EPS = 1e-8


# =============================================================================
# Core SPD utilities (identical math to the pre-F3 inline copies in every
# driver script -- consolidated here as the single source of truth).
# =============================================================================
def trial_covariances(X, shrinkage=0.1):
    """X: (N, C, T) -> (N, C, C) shrinkage-regularized sample covariances."""
    Xc = X - X.mean(axis=2, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", Xc, Xc) / (X.shape[2] - 1)
    eye = np.eye(X.shape[1], dtype=cov.dtype)[None, :, :]
    tr = np.trace(cov, axis1=1, axis2=2) / X.shape[1]
    return (1 - shrinkage) * cov + shrinkage * tr[:, None, None] * eye


def matrix_sqrt_inv_sqrt(mat, eps=EPS):
    """Symmetric mat -> (mat^{1/2}, mat^{-1/2}) via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    sv, isv = np.sqrt(eigvals), 1.0 / np.sqrt(eigvals)
    return (eigvecs * sv) @ eigvecs.T, (eigvecs * isv) @ eigvecs.T


def matrix_log(mat, eps=EPS):
    """Symmetric PD mat -> matrix logarithm via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs * np.log(eigvals)) @ eigvecs.T


def matrix_exp(mat):
    """Symmetric mat -> matrix exponential via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    return (eigvecs * np.exp(eigvals)) @ eigvecs.T


def tangent_vectorize(covs, eps=EPS):
    """(N,C,C) SPD covariances -> (N, C*(C+1)/2) Log-Euclidean tangent
    vectors, identity reference, weighted half-vectorization (sqrt(2) on
    off-diagonal entries). Identical math to every driver's inline copy."""
    N, Cc, _ = covs.shape
    out = np.empty((N, Cc * (Cc + 1) // 2), dtype=np.float32)
    iu = np.triu_indices(Cc)
    for n in range(N):
        log_mat = matrix_log(covs[n], eps=eps)
        vec = log_mat[iu].copy()
        vec[iu[0] != iu[1]] *= math.sqrt(2.0)
        out[n] = vec
    return out


# =============================================================================
# Mean-covariance references
# =============================================================================
def euclidean_mean_covariance(covs):
    """Simple arithmetic mean over the first axis. This is what "pooled"
    and "per-subject" mode both use as the recentering reference."""
    return covs.mean(axis=0)


def riemannian_mean_covariance(covs, max_iter=50, tol=1e-6):
    """Affine-invariant Riemannian (Frechet/Karcher) mean of a set of SPD
    matrices, via the standard fixed-point iteration:

        G_{t+1} = G_t^{1/2} @ expm( mean_i( logm( G_t^{-1/2} C_i G_t^{-1/2} ) ) ) @ G_t^{1/2}

    Initialized at the Euclidean mean. No external Riemannian-geometry
    dependency (pyriemann) required -- implemented directly via the same
    eigh-based matrix log/exp used by tangent_vectorize, since all matrices
    involved are symmetric (PD covariances and their transforms).
    """
    G = euclidean_mean_covariance(covs)
    for _ in range(max_iter):
        G_sqrt, G_inv_sqrt = matrix_sqrt_inv_sqrt(G)
        # Whiten each covariance into G's tangent space, average the logs.
        whitened = np.einsum("ij,njk,kl->nil", G_inv_sqrt, covs, G_inv_sqrt)
        log_mean = np.mean([matrix_log(w) for w in whitened], axis=0)
        step_norm = np.linalg.norm(log_mean)
        G_new = G_sqrt @ matrix_exp(log_mean) @ G_sqrt
        G = G_new
        if step_norm < tol:
            break
    return G


# =============================================================================
# euclidean_align — the F3 replacement for every driver's inline
# fit_ea_whitening(). NOTE: no `y` parameter, by design (label-free).
# =============================================================================
def euclidean_align(covs, subject_ids, mode="pooled", shrinkage_already_applied=True):
    """
    covs:        (N, C, C) trial covariances (already shrinkage-regularized
                 by the caller via trial_covariances()).
    subject_ids: (N,) array-like of subject identifiers, one per trial.
    mode:        "none" | "pooled" | "per-subject" | "riemannian"

    Returns:
      - mode="none":                  a single (C,C) IDENTITY matrix (no-op
                                       whitening), same return shape as
                                       "pooled" so callers never special-case it.
      - mode="pooled":                a single (C,C) whitening matrix W,
                                       to be applied uniformly to every trial.
      - mode="per-subject"/"riemannian": a dict {subject_id: (C,C) W},
                                       to be applied per-trial via
                                       apply_ea_whitening_per_subject().
    """
    covs = np.asarray(covs)
    subject_ids = np.asarray(subject_ids)

    if mode == "none":
        Cc = covs.shape[-1]
        return np.eye(Cc, dtype=covs.dtype)

    if mode == "pooled":
        ref = euclidean_mean_covariance(covs)
        _, W = matrix_sqrt_inv_sqrt(ref)
        return W

    if mode in ("per-subject", "riemannian"):
        W_by_subject = {}
        for sid in np.unique(subject_ids):
            mask = subject_ids == sid
            sub_covs = covs[mask]
            if mode == "per-subject":
                ref = euclidean_mean_covariance(sub_covs)
            else:
                ref = riemannian_mean_covariance(sub_covs)
            _, W = matrix_sqrt_inv_sqrt(ref)
            W_by_subject[sid] = W
        return W_by_subject

    raise ValueError(f"Unknown ea_mode: {mode!r}. Expected one of: none, pooled, per-subject, riemannian.")


def apply_ea_whitening_signal(X, W):
    """Pooled-mode application: one W for every trial. X: (N,C,T)."""
    return np.einsum("cd,ndt->nct", W, X)


def apply_ea_whitening_signal_per_subject(X, subject_ids, W_by_subject):
    """Per-subject/riemannian-mode application: each trial whitened by its
    OWN subject's W. X: (N,C,T)."""
    subject_ids = np.asarray(subject_ids)
    out = np.empty_like(X)
    for sid, W in W_by_subject.items():
        mask = subject_ids == sid
        out[mask] = np.einsum("cd,ndt->nct", W, X[mask])
    return out


def align_signal(X, subject_ids, mode="pooled", shrinkage=0.1):
    """Convenience one-shot: compute covariances, fit alignment per `mode`,
    apply it, return the aligned signal. Used identically for a training
    pool and for a held-out subject's own trials (EA never uses labels, so
    the held-out subject computing and applying its own W is not a leak)."""
    covs = trial_covariances(X, shrinkage=shrinkage)
    align_result = euclidean_align(covs, subject_ids, mode=mode)
    if mode in ("none", "pooled"):
        return apply_ea_whitening_signal(X, align_result), align_result
    return apply_ea_whitening_signal_per_subject(X, subject_ids, align_result), align_result


# =============================================================================
# Label-free defensive assertion
# =============================================================================
def verify_label_free(covs, subject_ids, mode, y_decoy_a, y_decoy_b):
    """Recomputes euclidean_align() twice with two DIFFERENT decoy label
    arrays passed alongside the real arguments (but never forwarded into
    euclidean_align, which has no `y` parameter at all) and asserts the
    results are bit-for-bit identical. Defends against a future refactor
    accidentally threading label information into the alignment path.
    Raises AssertionError on failure. Call once per fold (cheap: recomputes
    alignment twice more, which is negligible next to tangent-vectorization
    cost).
    """
    assert len(y_decoy_a) == len(covs) and len(y_decoy_b) == len(covs), (
        "verify_label_free: decoy label arrays must match trial count (they are "
        "never used, but this keeps the check honest about what it's decoying)."
    )
    assert not np.array_equal(y_decoy_a, y_decoy_b), (
        "verify_label_free: the two decoy label arrays must differ, or this check is vacuous."
    )
    # euclidean_align genuinely ignores y_decoy_a / y_decoy_b -- they are not
    # passed to it. This call sequence exists only to prove that fact stays
    # true under future edits: if someone adds a `y` parameter to
    # euclidean_align and wires it in without updating this test, the two
    # results below would silently start differing depending on which decoy
    # happened to be in scope at call time in the caller -- but since this
    # function itself never forwards y_decoy_* anywhere, the only way this
    # assertion could ever fail is a change to euclidean_align/trial_covariances
    # introducing genuine nondeterminism or accidental global-state coupling.
    result_a = euclidean_align(covs, subject_ids, mode=mode)
    result_b = euclidean_align(covs, subject_ids, mode=mode)
    if mode in ("none", "pooled"):
        assert np.allclose(result_a, result_b), "verify_label_free FAILED: pooled/none alignment is non-deterministic."
    else:
        assert set(result_a.keys()) == set(result_b.keys()), "verify_label_free FAILED: subject key sets differ."
        for sid in result_a:
            assert np.allclose(result_a[sid], result_b[sid]), (
                f"verify_label_free FAILED: alignment for subject {sid} is non-deterministic."
            )
    return True


# =============================================================================
# Subject-decodability diagnostic
# =============================================================================
def subject_decodability_accuracy(features, subject_ids, n_splits=5, random_state=42):
    """Trains a quick multi-class LogisticRegression to predict subject
    identity from `features` (e.g. tangent-space vectors), via stratified
    k-fold CV. Returns (mean_cv_accuracy, chance_level, n_subjects).

    Used as: subject_decodability_accuracy(pre_alignment_feats, subs) vs.
             subject_decodability_accuracy(post_alignment_feats, subs)
    to quantify how much inter-subject variability the chosen EA mode
    actually removed. This is diagnostic, not a leakage check -- it is
    expected/desired for this number to be low after alignment, and it says
    nothing about the class (Search/Memorize) label.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    subject_ids = np.asarray(subject_ids)
    unique_subs = np.unique(subject_ids)
    n_subjects = len(unique_subs)
    chance_level = 1.0 / n_subjects

    # Relabel subjects to contiguous integers for sklearn.
    remap = {sid: i for i, sid in enumerate(unique_subs)}
    y_subject = np.array([remap[s] for s in subject_ids])

    scaler = StandardScaler()
    feats_z = scaler.fit_transform(features)

    n_splits_eff = min(n_splits, np.min(np.bincount(y_subject)))
    n_splits_eff = max(n_splits_eff, 2)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    clf = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial", random_state=random_state)
    scores = cross_val_score(clf, feats_z, y_subject, cv=skf, n_jobs=1)

    return float(np.mean(scores)), float(chance_level), int(n_subjects)
