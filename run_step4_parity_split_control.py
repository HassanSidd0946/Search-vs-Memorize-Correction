# =============================================================================
# command to run:
#   modal run run_step4_parity_split_control.py::main
# run_step4_parity_split_control.py
#
# F-PARITY — PARITY-SPLIT CV + TEMPORAL-POSITION-DECODABILITY + WITHIN-BLOCK-
# POSITION BREAKDOWN (AUDIT.md Fix-ID table)
#
# WHY THIS SCRIPT EXISTS:
#   AUDIT.md Phase 0.5 Priority 2 found a real, unavoidable within-session
#   confound: this dataset is BLOCKED, not interleaved -- every Class-0
#   (Search) trial precedes every Class-1 (Memorize) trial in time, for
#   every subject (one transition at event #205 of 410). Class label is
#   therefore perfectly collinear with time-in-session within each subject.
#   The same investigation found the block order IS counterbalanced across
#   subjects on a clean parity pattern: odd-numbered subjects run
#   Search-first, even-numbered subjects run Memorize-first (checked via
#   .vmrk first-event-family for 10 subjects: sub-02,04,10,20,30 ->
#   Memorize-first; sub-03,05,09,15,25 -> Search-first).
#
#   That counterbalance makes a decisive control possible: if a classifier
#   trained only on odd-numbered subjects (where "early trial" tends to mean
#   Class 0) is evaluated on even-numbered subjects (where "early trial"
#   tends to mean Class 1), a classifier relying on a raw session-time-
#   position proxy rather than class-related neural signal should collapse
#   toward chance -- the "early=Class 0" shortcut is inverted in the test
#   population. Accuracy surviving this swap is evidence against a pure
#   session-position confound explaining the result.
#
#   This script runs that parity-split CV in both directions, and then
#   layers two cheap post-hoc analyses on the SAME tangent-space features
#   (no extra data, no extra Modal cost per AUDIT.md's own estimate):
#     1. Temporal-position-decodability: train a classifier to predict
#        WITHIN-BLOCK POSITION THIRD (early/mid/late, chronological, computed
#        separately per subject per class-block) instead of class label,
#        using the identical odd->even / even->odd split. High cross-
#        population decodability of pure session position would be a bad
#        sign for the class-decoding result's independence from timing.
#     2. Within-block-position (thirds) breakdown: slice the class-decoding
#        test accuracy from (1. above's parity-split CV) by which third of
#        its class-block each test trial came from, to check whether
#        accuracy is concentrated in trials closer to vs. further from the
#        Search<->Memorize transition.
#
#   Spatial feature pipeline (trial_covariances, fit_ea_whitening,
#   apply_ea_whitening_signal, tangent_vectorize -- raw identity-reference
#   tangent vectors, no group Frechet mean) is copied verbatim from
#   run_step4_matched_spatial_control.py so this control uses the exact same
#   feature representation as the headline results, per AUDIT.md's own
#   "reuses EA/tangent-space pipeline from run_step4_matched_spatial_
#   control.py" spec. Evaluation protocol is NOT LOSO here -- it is a
#   2-fold parity split (by design; that IS the control) with a single
#   pooled classifier per direction, no per-subject calibration step (there
#   is no natural held-out-subject calibration slice when the test set is
#   an entire parity-matched population at once).
#
# Usage: modal run run_step4_parity_split_control.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-parity-split-control")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_parity_split_control.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_parity_split_control():

    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-parity-split")

    np.random.seed(RANDOM_SEED)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")
    # F-SILENT hardening: a silently-failed subject upstream (e.g. sub-09's
    # truncated .eeg export, see AUDIT.md D2) must never produce a
    # clean-looking-but-incomplete run.
    assert len(np.unique(subjects_np)) == 29, (
        f"Expected exactly 29 subjects (30 - sub-09 exclusion per AUDIT.md D2), "
        f"got {len(np.unique(subjects_np))}: {sorted(np.unique(subjects_np).tolist())}"
    )

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    odd_subjects  = [s for s in unique_subjects if int(s) % 2 == 1]
    even_subjects = [s for s in unique_subjects if int(s) % 2 == 0]
    log.info(f"Odd subjects  (n={len(odd_subjects)}): {odd_subjects}")
    log.info(f"Even subjects (n={len(even_subjects)}): {even_subjects}")

    # =========================================================================
    # WITHIN-BLOCK POSITION THIRDS — computed once, globally, from raw trial
    # order. epochs.get_data() in run_data_engine_on_modal.py preserves
    # chronological event order and trials are never shuffled anywhere
    # upstream, so for each subject, the trials with y==0 appear in
    # chronological order among themselves (same for y==1). Splitting each
    # subject's per-class block into three equal chronological chunks gives
    # early/mid/late thirds relative to that block's own span.
    # =========================================================================
    def compute_within_block_thirds(y_np, subjects_np, unique_subjects):
        position_third = np.full(len(y_np), -1, dtype=np.int64)
        for sub in unique_subjects:
            sub_idx = np.where(subjects_np == sub)[0]
            for cls in (0, 1):
                cls_idx = sub_idx[y_np[sub_idx] == cls]
                n = len(cls_idx)
                if n == 0:
                    continue
                thirds = np.minimum((np.arange(n) * 3) // n, 2)
                position_third[cls_idx] = thirds
        assert (position_third >= 0).all(), "Every trial must be assigned a within-block position third."
        return position_third

    position_third_np = compute_within_block_thirds(y_np, subjects_np, unique_subjects)
    log.info(f"Within-block position thirds computed: "
             f"counts={np.bincount(position_third_np).tolist()} (early/mid/late)")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES — IDENTICAL to run_step4_matched_spatial_control.py
    # =========================================================================
    def trial_covariances(X, shrinkage=COV_SHRINKAGE):
        Xc = X - X.mean(axis=2, keepdims=True)
        cov = np.einsum("nct,ndt->ncd", Xc, Xc) / (X.shape[2] - 1)
        eye = np.eye(X.shape[1], dtype=cov.dtype)[None, :, :]
        tr = np.trace(cov, axis1=1, axis2=2) / X.shape[1]
        return (1 - shrinkage) * cov + shrinkage * tr[:, None, None] * eye

    def matrix_sqrt_inv_sqrt(mat, eps=1e-8):
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, eps, None)
        sv, isv = np.sqrt(eigvals), 1.0 / np.sqrt(eigvals)
        return (eigvecs * sv) @ eigvecs.T, (eigvecs * isv) @ eigvecs.T

    def fit_ea_whitening(X_train, shrinkage=COV_SHRINKAGE):
        covs = trial_covariances(X_train, shrinkage)
        _, W = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W

    def apply_ea_whitening_signal(X, W):
        return np.einsum("cd,ndt->nct", W, X)

    def tangent_vectorize(covs, eps=1e-8):
        """RAW tangent vectors, IDENTITY reference — no learned compression,
        no group Frechet mean, ever. Identical to the matched-spatial-control script."""
        N, Cc, _ = covs.shape
        out = np.empty((N, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N):
            eigvals, eigvecs = np.linalg.eigh(covs[n])
            eigvals = np.clip(eigvals, eps, None)
            log_mat = (eigvecs * np.log(eigvals)) @ eigvecs.T
            vec = log_mat[iu].copy()
            vec[iu[0] != iu[1]] *= math.sqrt(2.0)
            out[n] = vec
        return out

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    # C3 hardening (2026-08-20, per F-DRIFT-E's INVALID-DESIGN failure -- "all accuracies in
    # [0,1]" is vacuous against a base-rate artifact). Every classification result must report
    # class balance, majority-class rate, accuracy-minus-majority-rate, and balanced accuracy.
    # For BINARY results (n_classes=2, the default), fails loudly if balance falls outside 45/55
    # unless the caller explicitly declares an imbalanced design. The 45/55 gate does not
    # generalize cleanly to n_classes>2 (used here for the 3-way position-third-decodability
    # check, chance=1/3) -- multi-class calls report full diagnostics but are not gated.
    def c3_balance_check(y_true, y_pred, acc, label, declared_imbalanced_design=False, n_classes=2):
        counts = np.bincount(y_true, minlength=n_classes)
        n = len(y_true)
        balance = (counts / n).tolist()
        majority_rate = float(counts.max() / n)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        lift = float(acc - majority_rate)
        print(f"    [C3] {label}: class_balance={[round(b, 4) for b in balance]} "
              f"majority_rate={majority_rate:.4f} acc={acc:.4f} "
              f"acc_minus_majority={lift:+.4f} balanced_acc={bal_acc:.4f}")
        if n_classes == 2 and not declared_imbalanced_design:
            assert 0.45 <= min(balance) and max(balance) <= 0.55, (
                f"[C3 FAIL] {label}: class balance {balance} is outside the 45/55 band and this "
                "script does not declare an imbalanced design. Fix the design or set "
                "declared_imbalanced_design=True and treat balanced_accuracy as the primary metric."
            )
        return {"class_balance": balance, "majority_class_rate": majority_rate,
                "accuracy_minus_majority_rate": lift, "balanced_accuracy": bal_acc}

    # =========================================================================
    # ONE DIRECTION OF THE PARITY SPLIT: fit spatial pipeline on train_subjects,
    # evaluate class-decoding AND position-third-decoding on test_subjects.
    # =========================================================================
    def run_direction(direction_name, train_subjects, test_subjects):
        log.info(f"\n{'='*70}\n  DIRECTION: {direction_name}  "
                 f"(train n={len(train_subjects)} -> test n={len(test_subjects)})\n{'='*70}")
        dir_start = time.time()

        train_mask = np.isin(subjects_np, train_subjects)
        test_mask  = np.isin(subjects_np, test_subjects)
        X_train, y_train = X_np[train_mask], y_np[train_mask]
        X_test, y_test = X_np[test_mask], y_np[test_mask]
        subjects_test = subjects_np[test_mask]
        position_third_test = position_third_np[test_mask]

        mu = X_train.mean(axis=(0, 2), keepdims=True)
        sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train_z = ((X_train - mu) / sd).astype(np.float32)
        X_test_z = ((X_test - mu) / sd).astype(np.float32)

        W = fit_ea_whitening(X_train_z)
        X_train_aligned = apply_ea_whitening_signal(X_train_z, W).astype(np.float32)
        X_test_aligned = apply_ea_whitening_signal(X_test_z, W).astype(np.float32)

        tan_train = tangent_vectorize(trial_covariances(X_train_aligned))
        tan_test = tangent_vectorize(trial_covariances(X_test_aligned))

        scaler = StandardScaler()
        tan_train_z = scaler.fit_transform(tan_train)
        tan_test_z = scaler.transform(tan_test)

        n_components = min(PCA_MAX_COMPONENTS, tan_train_z.shape[1] - 1, tan_train_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(tan_train_z)
        X_test_pca = pca.transform(tan_test_z)

        # --- (1) CLASS-DECODING: the actual parity-split CV result -------------
        clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED)
        clf.fit(X_train_pca, y_train)
        class_preds = clf.predict(X_test_pca)
        pooled_class_acc = float(accuracy_score(y_test, class_preds))
        class_metrics = compute_binary_metrics(y_test, class_preds)
        class_plaus = c3_balance_check(y_test, class_preds, pooled_class_acc,
                                        f"{direction_name} class-decoding")

        per_subject_acc = {}
        for sub in test_subjects:
            sub_test_mask = subjects_test == sub
            if sub_test_mask.sum() == 0:
                continue
            per_subject_acc[sub] = float(accuracy_score(y_test[sub_test_mask], class_preds[sub_test_mask]))
        per_subject_acc_values = list(per_subject_acc.values())

        # --- (2) WITHIN-BLOCK-POSITION (THIRDS) BREAKDOWN, post-hoc on (1) -----
        thirds_breakdown = {}
        for third in (0, 1, 2):
            third_mask = position_third_test == third
            if third_mask.sum() == 0:
                continue
            thirds_breakdown[["early", "mid", "late"][third]] = {
                "n_trials": int(third_mask.sum()),
                "accuracy": float(accuracy_score(y_test[third_mask], class_preds[third_mask])),
            }

        # --- (3) TEMPORAL-POSITION-DECODABILITY: predict position-third instead
        #     of class, same train/test split, same PCA features. Chance = 1/3.
        pos_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED,
                                      multi_class="multinomial")
        position_third_train = position_third_np[train_mask]
        pos_clf.fit(X_train_pca, position_third_train)
        pos_preds = pos_clf.predict(X_test_pca)
        position_decodability_acc = float(accuracy_score(position_third_test, pos_preds))
        position_plaus = c3_balance_check(position_third_test, pos_preds, position_decodability_acc,
                                           f"{direction_name} position-third-decodability", n_classes=3)

        log.info(f"  Class-decoding pooled acc   : {pooled_class_acc:.4f}  "
                 f"(per-subject mean {np.mean(per_subject_acc_values):.4f} ± {np.std(per_subject_acc_values):.4f}, "
                 f"n_test_subjects={len(per_subject_acc_values)})")
        log.info(f"  Within-block thirds breakdown: {thirds_breakdown}")
        log.info(f"  Position-third decodability  : {position_decodability_acc:.4f} (chance = 0.3333)")
        log.info(f"  Direction elapsed: {time.time()-dir_start:.0f}s")

        return {
            "direction": direction_name,
            "train_subjects": train_subjects, "test_subjects": test_subjects,
            "n_train_trials": int(train_mask.sum()), "n_test_trials": int(test_mask.sum()),
            "tangent_dim": int(tan_train.shape[1]),
            "class_decoding": {
                "pooled_accuracy": pooled_class_acc,
                "per_subject_accuracy": per_subject_acc,
                "per_subject_mean": float(np.mean(per_subject_acc_values)),
                "per_subject_std": float(np.std(per_subject_acc_values)),
                "plausibility": class_plaus,
                **class_metrics,
            },
            "within_block_position_thirds_breakdown": thirds_breakdown,
            "position_third_decodability": {
                "accuracy": position_decodability_acc,
                "chance_level": 1.0 / 3.0,
                "plausibility": position_plaus,
            },
        }

    # =========================================================================
    # RUN BOTH DIRECTIONS
    # =========================================================================
    result_odd_to_even = run_direction("train_odd_test_even", odd_subjects, even_subjects)
    result_even_to_odd = run_direction("train_even_test_odd", even_subjects, odd_subjects)

    mean_pooled_class_acc = float(np.mean([
        result_odd_to_even["class_decoding"]["pooled_accuracy"],
        result_even_to_odd["class_decoding"]["pooled_accuracy"],
    ]))
    mean_position_decodability = float(np.mean([
        result_odd_to_even["position_third_decodability"]["accuracy"],
        result_even_to_odd["position_third_decodability"]["accuracy"],
    ]))

    log.info(f"\n{'='*70}\n  F-PARITY SUMMARY\n{'='*70}")
    log.info(f"  Mean pooled class-decoding acc across both directions : {mean_pooled_class_acc:.4f}")
    log.info(f"  Mean position-third decodability across both directions: {mean_position_decodability:.4f} "
             f"(chance = 0.3333)")

    results_payload = {
        "condition": "F-PARITY — parity-split CV (odd<->even subjects) + temporal-position-decodability "
                      "+ within-block-position thirds breakdown, identity-reference tangent, pool-only EA, no Mamba",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE,
        },
        "odd_subjects": odd_subjects, "even_subjects": even_subjects,
        "directions": [result_odd_to_even, result_even_to_odd],
        "mean_pooled_class_decoding_accuracy": mean_pooled_class_acc,
        "mean_position_third_decodability_accuracy": mean_position_decodability,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    # =========================================================================
    # C3 plausibility assertions -- printed next to the numbers, not silent.
    # Run AFTER the write above so a failing assertion never suppresses the
    # diagnostic artifact.
    # =========================================================================
    for res in (result_odd_to_even, result_even_to_odd):
        acc = res["class_decoding"]["pooled_accuracy"]
        pos_acc = res["position_third_decodability"]["accuracy"]
        assert 0.0 <= acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] {res['direction']} class-decoding accuracy {acc} outside [0,1]"
        assert 0.0 <= pos_acc <= 1.0, f"[C3 PLAUSIBILITY FAIL] {res['direction']} position-third accuracy {pos_acc} outside [0,1]"
    assert len(odd_subjects) + len(even_subjects) == 29, (
        f"[C3 PLAUSIBILITY FAIL] expected 29 total subjects across odd+even, "
        f"got {len(odd_subjects)}+{len(even_subjects)}={len(odd_subjects)+len(even_subjects)}"
    )
    assert (len(odd_subjects), len(even_subjects)) == (14, 15), (
        f"[C3 PLAUSIBILITY FAIL] expected 14 odd / 15 even subjects (DECISIONS.md D2's documented "
        f"14/15 split after sub-09 exclusion), got {len(odd_subjects)} odd / {len(even_subjects)} even"
    )
    log.info(f"  [C3] plausibility: 14/15 odd/even split confirmed, both directions' accuracies in [0,1] -- OK")

    return {
        "mean_pooled_class_decoding_accuracy": mean_pooled_class_acc,
        "mean_position_third_decodability_accuracy": mean_position_decodability,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-PARITY — Parity-split CV (odd<->even subjects) + temporal-position-")
    print("decodability + within-block-position thirds breakdown\n")
    results = run_parity_split_control.remote()
    print("\nF-PARITY RESULTS:")
    for k, v in results.items():
        print(f"  {k:<40}: {v}")
