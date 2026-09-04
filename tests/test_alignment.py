"""
tests/test_alignment.py

F3 -- unit tests for eeg_alignment.py, run entirely on synthetic data (no
Modal, no dataset dependency -- these run locally and fast).

Usage: python -m pytest tests/test_alignment.py -v
       (or: python tests/test_alignment.py, runs a plain assert-based
       fallback if pytest is unavailable)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import eeg_alignment as ea

RNG = np.random.RandomState(0)


def _make_synthetic_signal(n_subjects=6, n_trials_per_subject=20, n_channels=8, n_times=100, seed=0,
                            shared_mixing=True):
    """Synthetic multi-subject EEG-like data where each subject has a
    DIFFERENT covariance scale (simulating inter-subject amplitude/
    impedance variability that EA is meant to remove).

    shared_mixing=True: all subjects share the SAME underlying spatial
    mixing pattern, differing only in scale -- this isolates exactly the
    variability EA is designed to remove, for tests that check EA's effect
    on subject decodability. shared_mixing=False: each subject gets an
    independent random mixing matrix (used for shape/API tests only, where
    inter-subject spatial-pattern differences are irrelevant)."""
    rng = np.random.RandomState(seed)
    X_list, subject_ids = [], []
    shared_matrix = rng.normal(size=(n_channels, n_channels))
    for s in range(n_subjects):
        scale = 0.5 + s * 0.4  # deliberately different per subject
        mixing = shared_matrix if shared_mixing else rng.normal(size=(n_channels, n_channels))
        for _ in range(n_trials_per_subject):
            src = rng.normal(size=(n_channels, n_times)) * scale
            trial = mixing @ src
            X_list.append(trial)
            subject_ids.append(s)
    X = np.stack(X_list).astype(np.float32)
    subject_ids = np.array(subject_ids)
    return X, subject_ids


def test_pooled_mode_shape():
    X, subs = _make_synthetic_signal()
    covs = ea.trial_covariances(X)
    W = ea.euclidean_align(covs, subs, mode="pooled")
    assert W.shape == (X.shape[1], X.shape[1])
    X_aligned = ea.apply_ea_whitening_signal(X, W)
    assert X_aligned.shape == X.shape


def test_none_mode_is_identity_noop():
    """F7: '--ea-mode none' must return an identity matrix and leave the
    signal numerically unchanged -- it exists so drivers can loop over the
    2x2 alignment/representation grid's no-EA cells via the same flag as
    every aligned mode, without a separate code path."""
    X, subs = _make_synthetic_signal()
    covs = ea.trial_covariances(X)
    W_none = ea.euclidean_align(covs, subs, mode="none")
    assert W_none.shape == (X.shape[1], X.shape[1])
    assert np.allclose(W_none, np.eye(X.shape[1]))
    X_aligned = ea.apply_ea_whitening_signal(X, W_none)
    assert np.allclose(X_aligned, X)


def test_per_subject_mode_returns_dict():
    X, subs = _make_synthetic_signal()
    covs = ea.trial_covariances(X)
    W_by_sub = ea.euclidean_align(covs, subs, mode="per-subject")
    assert isinstance(W_by_sub, dict)
    assert set(W_by_sub.keys()) == set(np.unique(subs).tolist())
    X_aligned = ea.apply_ea_whitening_signal_per_subject(X, subs, W_by_sub)
    assert X_aligned.shape == X.shape


def test_riemannian_mode_returns_dict_and_is_close_to_euclidean_for_low_dispersion():
    X, subs = _make_synthetic_signal(n_channels=5, n_trials_per_subject=30)
    covs = ea.trial_covariances(X)
    W_euclid = ea.euclidean_align(covs, subs, mode="per-subject")
    W_riem = ea.euclidean_align(covs, subs, mode="riemannian")
    assert isinstance(W_riem, dict)
    assert set(W_riem.keys()) == set(W_euclid.keys())
    # Riemannian and Euclidean means of a tight cluster of SPD matrices
    # should be reasonably close (not identical, but same order of magnitude).
    for sid in W_euclid:
        diff = np.linalg.norm(W_riem[sid] - W_euclid[sid])
        scale = np.linalg.norm(W_euclid[sid])
        assert diff / (scale + 1e-8) < 2.0, f"riemannian/euclidean W diverged too much for subject {sid}"


def test_per_subject_alignment_equalizes_covariance_scale():
    """The whole point of per-subject EA: after alignment, each subject's
    OWN average trial covariance trace should be close to n_channels
    (identity-scale), removing the deliberately-injected per-subject scale
    difference."""
    X, subs = _make_synthetic_signal(n_channels=6, n_trials_per_subject=25)
    covs = ea.trial_covariances(X)
    W_by_sub = ea.euclidean_align(covs, subs, mode="per-subject")
    X_aligned = ea.apply_ea_whitening_signal_per_subject(X, subs, W_by_sub)
    covs_aligned = ea.trial_covariances(X_aligned)

    traces_by_subject = []
    for sid in np.unique(subs):
        mask = subs == sid
        traces_by_subject.append(np.trace(covs_aligned[mask].mean(axis=0)))
    traces_by_subject = np.array(traces_by_subject)
    # Pre-alignment traces varied by construction (scale 0.5..2.5, squared
    # effect on covariance -> large spread). Post-alignment should be tight.
    assert traces_by_subject.std() / traces_by_subject.mean() < 0.15, (
        f"Per-subject EA did not equalize covariance scale: traces={traces_by_subject}"
    )


def test_verify_label_free_passes():
    X, subs = _make_synthetic_signal()
    covs = ea.trial_covariances(X)
    y_a = RNG.randint(0, 2, size=len(subs))
    y_b = 1 - y_a
    assert ea.verify_label_free(covs, subs, mode="pooled", y_decoy_a=y_a, y_decoy_b=y_b)
    assert ea.verify_label_free(covs, subs, mode="per-subject", y_decoy_a=y_a, y_decoy_b=y_b)


def test_subject_decodability_drops_after_alignment():
    X, subs = _make_synthetic_signal(n_channels=6, n_trials_per_subject=25)
    covs_pre = ea.trial_covariances(X)
    tan_pre = ea.tangent_vectorize(covs_pre)

    W_by_sub = ea.euclidean_align(covs_pre, subs, mode="per-subject")
    X_aligned = ea.apply_ea_whitening_signal_per_subject(X, subs, W_by_sub)
    covs_post = ea.trial_covariances(X_aligned)
    tan_post = ea.tangent_vectorize(covs_post)

    acc_pre, chance, n_subjects = ea.subject_decodability_accuracy(tan_pre, subs, n_splits=3)
    acc_post, _, _ = ea.subject_decodability_accuracy(tan_post, subs, n_splits=3)

    assert n_subjects == 6
    assert 0.0 < chance < 1.0
    # Pre-alignment, the deliberately-injected per-subject scale difference
    # should make subjects clearly decodable (well above the 1/6 chance
    # level -- finite-sample covariance-estimation noise at n_times=100
    # keeps this well short of "near-perfect", so we require "clearly
    # above chance" rather than an arbitrary near-1.0 bar).
    assert acc_pre > 3 * chance, f"Expected pre-alignment subject decodability well above chance, got {acc_pre} (chance={chance})"
    # Post-alignment, decodability should drop to (or below) chance -- EA
    # should remove essentially all of the injected per-subject scale signal.
    assert acc_post < acc_pre, f"Expected alignment to reduce subject decodability: pre={acc_pre} post={acc_post}"
    assert acc_post < 1.5 * chance, f"Expected post-alignment subject decodability near chance, got {acc_post} (chance={chance})"


def _run_all():
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in fns:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
