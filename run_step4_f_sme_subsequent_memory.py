# =============================================================================
# command to run:
#   modal run run_step4_f_sme_subsequent_memory.py::main
# run_step4_f_sme_subsequent_memory.py
#
# F-SME — SUBSEQUENT MEMORY (REMEMBERED VS. FORGOTTEN), THE LAST EXPERIMENT
# (new fix-ID, added 2026-08-20, AUDIT.md fix register; DECISIONS.md
# pre-registered rule) -- FINAL EXPERIMENT, UNCONDITIONALLY.
#
# WHY THIS SCRIPT EXISTS:
#   F-DRIFT-G (ledger L031) found the real Search-vs-Memorize contrast
#   SURVIVES onset (rest-break) exclusion at post_cal_balanced=0.7215 --
#   but that result was immediately flagged as a DISCLOSED RULE
#   MIS-SPECIFICATION: "survives onset exclusion" was equated with
#   "genuine task signal," which is not a valid inference. The pseudo-
#   label contrasts that collapsed under onset exclusion (F-DRIFT-F(a))
#   were WITHIN-block; the real Search-vs-Memorize contrast is BETWEEN
#   blocks. Removing trials from inside each block does not remove the
#   block-level confound: task instruction, stimulus set (disjoint within
#   subject -- each scene/object is shown to a given subject exactly
#   once, under exactly one condition), session half, and post-break
#   state are all still perfectly collinear with class. F-DRIFT-G must
#   NOT be written up as evidence of task decoding.
#
#   Subsequent memory (remembered vs. forgotten, the classic Dm-effect /
#   subsequent-memory-effect paradigm) is the ONLY contrast available in
#   ds005189 that is simultaneously WITHIN-block, WITHIN-task, WITHIN-
#   stimulus-set, and WITHIN-session-half -- i.e. the only contrast the
#   block confound (or the rest-break discontinuity, or the F-STIM
#   stimulus-identity confound) cannot explain, because both classes
#   (later-remembered, later-forgotten) are drawn from the SAME task
#   block, interleaved throughout it, matched on encoding condition.
#
# LINKING METHODOLOGY (established and verified BEFORE this script was
# written -- see AUDIT.md's Phase 0.5 Priority 1 and the 2026-08-20
# verification pass):
#   - `(scene, obj)` uniquely links each of a subject's 100 Encode trials
#     (`*_task-SearchSupRecFamEncode_beh.tsv`, 50 Search + 50
#     Memorization, in chronological order) to exactly one `Target` row
#     in that subject's Test/retrieval file
#     (`*_task-SearchSupRecFamTest_beh.tsv`, 300 rows: 100 Target/50
#     Distractor/150 New(Lure)). Verified 0 unmatched, 0 collisions, on
#     every subject checked.
#   - Outcome: `rk_judg` in {Remember, Know} -> REMEMBERED (label 0);
#     `rk_judg` == New (a miss on a real studied item) -> FORGOTTEN
#     (label 1, the minority class).
#   - EEG-epoch-to-behavioral-trial correspondence: raw marker code
#     "Stimulus/ 11" (within the Search block) and "Stimulus/ 21" (within
#     the Memorization block) each fire EXACTLY 50 times per subject, in
#     strictly chronological (monotonically increasing sample-position)
#     order -- verified against every locally-available subject's raw
#     `.vmrk` file before this script was written. This is the ONLY
#     marker-code pair in the raw stream confirmed to correspond 1:1,
#     in-order, with the 50 behavioral trial rows of each task; the
#     other codes (10/12/13, 20/22/23, 101/201) do not have a verified
#     trial-level correspondence and are deliberately NOT used here --
#     using them would risk silently mislabeling subsequent-memory
#     outcome, which this script treats as unacceptable. This means
#     F-SME uses 50 epochs/block/subject (one per behavioral trial), not
#     the ~200/block used by the rest of the F-DRIFT family (which never
#     needed trial-level correspondence, only class-level).
#
# TEST DESIGN (pre-registered in DECISIONS.md BEFORE this script runs):
#   Label = subsequent memory outcome (0=remembered, 1=forgotten). Run
#   BOTH within Search-only trials, within Memorize-only trials, AND
#   pooled (both tasks together), identical calibrated pipeline
#   (EA/tangent/PCA/shrinkage-calibration, matching the rest of the
#   pre-F3 F-DRIFT family), single seed=42, full LOSO (fold count =
#   however many subjects survive the exclusion criterion below, NOT
#   necessarily 29).
#
#   AUC IS THE PRIMARY METRIC (roc_auc_score on the continuous decision
#   score) -- minority class is ~0-18% per subject/condition, so raw
#   accuracy is meaningless. Also reports balanced accuracy and the
#   per-subject minority-class (forgotten) trial count.
#
#   EXCLUSION (computed and logged BEFORE any classifier runs, per
#   condition independently): any subject with fewer than
#   MINORITY_MIN_TRIALS=10 forgotten trials in that condition is
#   excluded from that condition's LOSO entirely (not just from being a
#   test fold) -- exactly who and why is reported before results.
#
# PRE-REGISTERED INTERPRETATION RULE (DECISIONS.md, fixed BEFORE this
# script runs, stated as the inference each threshold supports, not just
# the threshold):
#   AUC >= 0.60 with 95% CI excluding 0.5
#       -> genuine subsequent-memory signal, free of the block confound.
#          A real cognitive result, underpowered but clean.
#   AUC 0.55-0.60, CI excluding 0.5
#       -> weak but present; report as exploratory with power caveats.
#   AUC <= 0.55, or CI containing 0.5
#       -> no detectable confound-free cognitive signal in this dataset;
#          the methodological paper is the paper. A null here is NOT
#          evidence against the effect existing -- it is a power
#          statement given 0-18% minority trials per subject.
#
# Usage: modal run run_step4_f_sme_subsequent_memory.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-f-sme-subsequent-memory")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_PATH   = "/data"
OUTPUT_JSON   = "/data/results_condition4_f_sme_subsequent_memory.json"

NUM_SUBJECTS_TOTAL = 30
SUB09_EXCLUDED      = "09"   # truncated raw EEG file, per D2 -- same exclusion as every other script

SFREQ_RESAMPLE = 250
TMIN, TMAX, BASELINE = -0.2, 0.8, (None, 0)
L_FREQ, H_FREQ = 1.0, 40.0

# ONLY these two codes are used -- see header for why (verified 1:1,
# in-order correspondence with the 50 behavioral trial rows; every other
# code lacks a verified trial-level mapping).
EVENT_ID_TASK = {"Stimulus/ 11": 0, "Stimulus/ 21": 1}   # 0=Search block, 1=Memorization block

RANDOM_SEED         = 42
COV_SHRINKAGE       = 0.1
PCA_MAX_COMPONENTS  = 35
LOGREG_C            = 1.0
CAL_FRACTION        = 0.15
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

MINORITY_MIN_TRIALS = 10

# DECISIONS.md pre-registered thresholds (fixed before this script runs).
AUC_GENUINE_THRESHOLD = 0.60
AUC_WEAK_THRESHOLD    = 0.55

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2", "scikit-learn==1.4.2", "scipy==1.14.1",
        "mne==1.7.1", "openneuro-py==2024.2.0",
    )
)


# =============================================================================
# STEP A: per-subject raw extraction + behavioral linkage -> (X, sme_label,
# task_label) for the 50+50 code-11/code-21 epochs. Separate Modal function
# so it can be retried/inspected independently of the LOSO stage.
# =============================================================================
@app.function(image=image, cpu=2.0, volumes={VOLUME_PATH: volume}, timeout=3600, memory=8192)
def extract_subject_sme(sub_idx: int):
    import os, glob, csv, logging
    import numpy as np
    import mne
    import openneuro

    mne.set_log_level("WARNING")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f-sme-extract")

    sub_id = f"{sub_idx:02d}"
    sub_tag = f"sub-{sub_id}"
    raw_dir = os.path.join(VOLUME_PATH, "openneuro", sub_tag)

    openneuro.download(dataset="ds005189", target_dir=os.path.join(VOLUME_PATH, "openneuro"),
                        include=[sub_tag])

    vhdr_files = glob.glob(os.path.join(raw_dir, "**", "*.vhdr"), recursive=True)
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr found under {raw_dir}")
    raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True, verbose=False)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir", verbose=False)
    raw.resample(sfreq=SFREQ_RESAMPLE, verbose=False)

    events, event_id_found = mne.events_from_annotations(raw, event_id=EVENT_ID_TASK, verbose=False)
    if len(events) == 0:
        raise ValueError(f"{sub_tag}: zero code-11/21 events found")
    epochs = mne.Epochs(raw, events, event_id=event_id_found, tmin=TMIN, tmax=TMAX,
                         baseline=BASELINE, preload=True, reject=None, verbose=False)
    X_sub = epochs.get_data(copy=True).astype(np.float32)
    inv_event_map = {event_id_found[k]: v for k, v in EVENT_ID_TASK.items() if k in event_id_found}
    task_label_sub = np.array([inv_event_map[c] for c in epochs.events[:, 2]], dtype=np.int64)

    n_search = int((task_label_sub == 0).sum())
    n_memorize = int((task_label_sub == 1).sum())
    assert n_search == 50, (
        f"{sub_tag}: expected exactly 50 code-11 (Search) epochs, got {n_search} -- the verified "
        "1:1 trial correspondence this script depends on does not hold for this subject; HALT "
        "rather than silently mislabel subsequent-memory outcome."
    )
    assert n_memorize == 50, (
        f"{sub_tag}: expected exactly 50 code-21 (Memorization) epochs, got {n_memorize} -- same "
        "halt-on-violation rationale as above."
    )

    # ---- Behavioral linkage: (scene, obj) -> rk_judg, in chronological order ----
    beh_dir = os.path.join(raw_dir, "beh")
    encode_path = glob.glob(os.path.join(beh_dir, "*Encode_beh.tsv"))
    test_path = glob.glob(os.path.join(beh_dir, "*Test_beh.tsv"))
    assert len(encode_path) == 1 and len(test_path) == 1, (
        f"{sub_tag}: expected exactly one Encode and one Test behavioral file, "
        f"found {encode_path}, {test_path}"
    )
    with open(encode_path[0]) as f:
        encode_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(test_path[0]) as f:
        test_rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(encode_rows) == 100, f"{sub_tag}: expected 100 Encode rows, got {len(encode_rows)}"

    target_map = {(r["scene"], r["obj"]): r for r in test_rows if r["obj_type"] == "Target"}
    assert len(target_map) == 100, f"{sub_tag}: expected 100 Target rows in Test file, got {len(target_map)}"

    search_rows = [r for r in encode_rows if r["task"] == "Search"]
    memorize_rows = [r for r in encode_rows if r["task"] == "Memorization"]
    assert len(search_rows) == 50 and len(memorize_rows) == 50, (
        f"{sub_tag}: Encode file task split is not 50/50 (got {len(search_rows)}/{len(memorize_rows)})"
    )

    def outcome_labels(behavioral_rows):
        labels = []
        for r in behavioral_rows:
            key = (r["scene"], r["obj"])
            assert key in target_map, f"{sub_tag}: Encode row {key} has no matching Test Target row"
            rk = target_map[key]["rk_judg"]
            labels.append(1 if rk == "New" else 0)   # 1 = forgotten (minority), 0 = remembered
        return np.array(labels, dtype=np.int64)

    sme_search = outcome_labels(search_rows)      # length 50, chronological order = code-11 order
    sme_memorize = outcome_labels(memorize_rows)  # length 50, chronological order = code-21 order

    # epochs.events preserves chronological (recording) order within each
    # code; task_label_sub tells us which code each epoch came from, but
    # NOT which is 1st/2nd/... within that code -- reconstruct per-code
    # running index to assign the correctly-ordered behavioral outcome.
    sme_label_sub = np.empty(len(task_label_sub), dtype=np.int64)
    running = {0: 0, 1: 0}
    for i, t in enumerate(task_label_sub):
        j = running[t]
        sme_label_sub[i] = sme_search[j] if t == 0 else sme_memorize[j]
        running[t] += 1
    assert running[0] == 50 and running[1] == 50

    n_forgotten_search = int(sme_search.sum())
    n_forgotten_memorize = int(sme_memorize.sum())
    log.info(f"{sub_tag}: n_forgotten_search={n_forgotten_search}/50, "
             f"n_forgotten_memorize={n_forgotten_memorize}/50, "
             f"n_forgotten_pooled={n_forgotten_search + n_forgotten_memorize}/100")

    return {
        "sub_id": sub_id, "X": X_sub, "task_label": task_label_sub, "sme_label": sme_label_sub,
        "n_forgotten_search": n_forgotten_search, "n_forgotten_memorize": n_forgotten_memorize,
    }


# =============================================================================
# STEP B: pool all subjects, compute exclusion, run calibrated LOSO with
# AUC as the primary metric, for each of {within_search, within_memorize,
# pooled}.
# =============================================================================
@app.function(image=image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=86400, memory=16384)
def run_sme_loso():
    import logging, time, math, json
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("f-sme-loso")
    np.random.seed(RANDOM_SEED)

    subject_indices = [i for i in range(1, NUM_SUBJECTS_TOTAL + 1) if f"{i:02d}" != SUB09_EXCLUDED]
    log.info(f"Extracting {len(subject_indices)} subjects (code-11/21 epochs + behavioral linkage)...")
    per_subject = {}
    for r in extract_subject_sme.map(subject_indices):
        per_subject[r["sub_id"]] = r
    assert len(per_subject) == 29, f"Expected 29 subjects extracted, got {len(per_subject)}"

    X_all = np.concatenate([per_subject[s]["X"] for s in per_subject], axis=0)
    task_all = np.concatenate([per_subject[s]["task_label"] for s in per_subject], axis=0)
    sme_all = np.concatenate([per_subject[s]["sme_label"] for s in per_subject], axis=0)
    subjects_all = np.concatenate(
        [np.full(len(per_subject[s]["task_label"]), s) for s in per_subject], axis=0)
    log.info(f"Pooled dataset: N={len(sme_all)} (29 subjects x 100 trials)")

    # =========================================================================
    # EXCLUSION -- computed and logged BEFORE any classifier runs, per
    # condition independently.
    # =========================================================================
    exclusion = {"within_search": [], "within_memorize": [], "pooled": []}
    included = {"within_search": [], "within_memorize": [], "pooled": []}
    for s in sorted(per_subject.keys()):
        r = per_subject[s]
        n_pooled = r["n_forgotten_search"] + r["n_forgotten_memorize"]
        for cond, n_minority in (("within_search", r["n_forgotten_search"]),
                                  ("within_memorize", r["n_forgotten_memorize"]),
                                  ("pooled", n_pooled)):
            if n_minority < MINORITY_MIN_TRIALS:
                exclusion[cond].append({"sub": s, "n_forgotten": n_minority,
                                         "reason": f"< {MINORITY_MIN_TRIALS} forgotten trials"})
            else:
                included[cond].append(s)

    log.info(f"\n{'='*70}\n  EXCLUSION (computed BEFORE any classification)\n{'='*70}")
    for cond in exclusion:
        log.info(f"  {cond}: {len(included[cond])} included, {len(exclusion[cond])} excluded")
        for e in exclusion[cond]:
            log.info(f"    EXCLUDED sub-{e['sub']}: n_forgotten={e['n_forgotten']} ({e['reason']})")

    # =========================================================================
    # RIEMANNIAN / EA / CLASSIFIER UTILITIES -- IDENTICAL to the rest of the
    # pre-F3 F-DRIFT family.
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
        _, W_ = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W_

    def apply_ea_whitening_signal(X, W_):
        return np.einsum("cd,ndt->nct", W_, X)

    def tangent_vectorize(covs, eps=1e-8):
        N_, Cc, _ = covs.shape
        out = np.empty((N_, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N_):
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

    # C3 hardening -- SME is a KNOWN, DECLARED imbalanced design (subsequent
    # memory, minority ~0-18%), so declared_imbalanced_design=True
    # throughout; AUC and balanced accuracy (not raw accuracy) are the
    # metrics that matter here.
    def c3_report(y_true, y_pred, score, acc, label):
        counts = np.bincount(y_true, minlength=2)
        n = len(y_true)
        balance = (counts / n).tolist()
        majority_rate = float(counts.max() / n)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")
        print(f"    [C3] {label}: class_balance={[round(b, 4) for b in balance]} "
              f"majority_rate={majority_rate:.4f} acc={acc:.4f} balanced_acc={bal_acc:.4f} "
              f"auc={auc:.4f} [DECLARED IMBALANCED]")
        return {"class_balance": balance, "majority_class_rate": majority_rate,
                "balanced_accuracy": bal_acc, "auc": auc}

    def linear_predict(coef, intercept, X):
        score = (X @ coef.T + intercept).ravel()
        return (score > 0).astype(int), score

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, seed):
        global_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed,
                                         class_weight="balanced").fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed,
                                             class_weight="balanced").fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=seed,
                                             class_weight="balanced").fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                preds_b, _ = linear_predict(coef_b, icpt_b, X_val)
                shrink_scores[shrink].append((preds_b == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # LOSO for ONE condition (within_search / within_memorize / pooled),
    # restricted to the subjects that pass THAT condition's exclusion.
    # =========================================================================
    def run_condition(cond_name, task_filter, eligible_subjects):
        log.info(f"\n{'='*70}\n  F-SME: {cond_name} ({len(eligible_subjects)} eligible subjects)\n{'='*70}")
        if task_filter is None:
            mask = np.isin(subjects_all, eligible_subjects)
        else:
            mask = np.isin(subjects_all, eligible_subjects) & (task_all == task_filter)
        X_cond, y_cond, subj_cond = X_all[mask], sme_all[mask], subjects_all[mask]

        fold_records = []
        pre_cal_accs, post_cal_accs = [], []
        pre_cal_aucs, post_cal_aucs = [], []

        for fold_idx, test_sub in enumerate(eligible_subjects):
            fold_start = time.time()
            is_holdout = subj_cond == test_sub
            X_train, y_train = X_cond[~is_holdout], y_cond[~is_holdout]
            X_k, y_k = X_cond[is_holdout], y_cond[is_holdout]
            assert len(np.unique(y_train)) == 2, f"{cond_name} fold sub-{test_sub}: training pool missing a class"
            if len(np.unique(y_k)) < 2:
                log.warning(f"  [{cond_name}] fold sub-{test_sub}: held-out subject has only one "
                            f"class present ({np.bincount(y_k, minlength=2).tolist()}) -- AUC "
                            "undefined for this fold, skipping (not silently averaged as 0.5).")
                continue

            mu = X_train.mean(axis=(0, 2), keepdims=True)
            sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train_z = ((X_train - mu) / sd).astype(np.float32)
            X_k_z = ((X_k - mu) / sd).astype(np.float32)

            W_ea = fit_ea_whitening(X_train_z)
            X_train_aligned = apply_ea_whitening_signal(X_train_z, W_ea).astype(np.float32)
            X_k_aligned = apply_ea_whitening_signal(X_k_z, W_ea).astype(np.float32)

            tan_train = tangent_vectorize(trial_covariances(X_train_aligned))
            tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
            tangent_dim = tan_train.shape[1]

            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
            cal_idx, test_idx = next(sss.split(tan_k, y_k))
            feat_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
            feat_test, y_test = tan_k[test_idx], y_k[test_idx]
            if len(np.unique(y_test)) < 2:
                log.warning(f"  [{cond_name}] fold sub-{test_sub}: scored 85% split lost the minority "
                            "class -- AUC undefined, skipping fold.")
                continue

            scaler = StandardScaler()
            feat_train_z = scaler.fit_transform(tan_train)
            feat_cal_z = scaler.transform(feat_cal)
            feat_test_z = scaler.transform(feat_test)

            n_components = min(PCA_MAX_COMPONENTS, feat_train_z.shape[1] - 1, feat_train_z.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_train_pca = pca.fit_transform(feat_train_z)
            X_cal_pca = pca.transform(feat_cal_z)
            X_test_pca = pca.transform(feat_test_z)

            coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
                X_train_pca, y_train, X_cal_pca, y_cal, RANDOM_SEED)
            pre_cal_preds = global_clf.predict(X_test_pca)
            pre_cal_score = global_clf.decision_function(X_test_pca)
            pre_cal_acc = float((pre_cal_preds == y_test).mean())
            final_preds, final_score = linear_predict(coef_final, icpt_final, X_test_pca)
            post_cal_acc = float((final_preds == y_test).mean())
            metrics = compute_binary_metrics(y_test, final_preds)
            pre_report = c3_report(y_test, pre_cal_preds, pre_cal_score, pre_cal_acc,
                                    f"{cond_name} fold sub-{test_sub} pre_cal")
            post_report = c3_report(y_test, final_preds, final_score, post_cal_acc,
                                     f"{cond_name} fold sub-{test_sub} post_cal")

            log.info(f"  [{cond_name}] fold {fold_idx+1}/{len(eligible_subjects)} sub-{test_sub} -> "
                     f"pre_cal_auc={pre_report['auc']:.4f} post_cal_auc={post_report['auc']:.4f} "
                     f"n_minority_test={int(y_test.sum())} [{time.time()-fold_start:.0f}s]")

            fold_records.append({
                "fold_index": fold_idx, "test_subject": str(test_sub),
                "tangent_dim": int(tangent_dim), "best_shrink_weight": float(best_shrink),
                "n_minority_total": int(y_k.sum()), "n_minority_test": int(y_test.sum()),
                "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": post_cal_acc,
                "pre_cal_report": pre_report, "post_cal_report": post_report,
                **metrics,
            })
            pre_cal_accs.append(pre_cal_acc); post_cal_accs.append(post_cal_acc)
            pre_cal_aucs.append(pre_report["auc"]); post_cal_aucs.append(post_report["auc"])

        n_scored_folds = len(fold_records)
        if n_scored_folds == 0:
            log.warning(f"  [{cond_name}] NO FOLDS SCORED (every eligible subject's held-out set "
                        "was single-class) -- condition is unevaluable, reporting empty.")
            return {"condition": cond_name, "n_eligible_subjects": len(eligible_subjects),
                    "n_scored_folds": 0, "fold_results": []}

        pre_auc_mean, post_auc_mean = float(np.mean(pre_cal_aucs)), float(np.mean(post_cal_aucs))
        pre_bal_mean = float(np.mean([r["pre_cal_report"]["balanced_accuracy"] for r in fold_records]))
        post_bal_mean = float(np.mean([r["post_cal_report"]["balanced_accuracy"] for r in fold_records]))

        # Bootstrap 95% CI over subjects (resample scored folds with
        # replacement, N=2000) -- same convention as scripts/drift_c_posthoc_analysis.py.
        rng = np.random.default_rng(20260820)
        boot_pre, boot_post = [], []
        for _ in range(2000):
            idx = rng.integers(0, n_scored_folds, size=n_scored_folds)
            boot_pre.append(np.mean([pre_cal_aucs[i] for i in idx]))
            boot_post.append(np.mean([post_cal_aucs[i] for i in idx]))
        pre_ci = [float(np.percentile(boot_pre, 2.5)), float(np.percentile(boot_pre, 97.5))]
        post_ci = [float(np.percentile(boot_post, 2.5)), float(np.percentile(boot_post, 97.5))]

        log.info(f"  [{cond_name}] DONE ({n_scored_folds}/{len(eligible_subjects)} folds scored) -- "
                 f"pre_cal_auc={pre_auc_mean:.4f} (95% CI {pre_ci}) "
                 f"post_cal_auc={post_auc_mean:.4f} (95% CI {post_ci})")

        return {
            "condition": cond_name, "n_eligible_subjects": len(eligible_subjects),
            "n_scored_folds": n_scored_folds,
            "fold_results": fold_records,
            "pre_calibration_auc_mean": pre_auc_mean, "pre_calibration_auc_95ci": pre_ci,
            "post_calibration_auc_mean": post_auc_mean, "post_calibration_auc_95ci": post_ci,
            "pre_calibration_balanced_accuracy_mean": pre_bal_mean,
            "post_calibration_balanced_accuracy_mean": post_bal_mean,
        }

    result_search = run_condition("within_search", 0, included["within_search"])
    result_memorize = run_condition("within_memorize", 1, included["within_memorize"])
    result_pooled = run_condition("pooled", None, included["pooled"])

    # =========================================================================
    # DECISIONS.md's pre-registered F-SME verdict, applied per condition to
    # POST-calibration AUC (the primary metric).
    # =========================================================================
    def apply_verdict(result):
        if result["n_scored_folds"] == 0:
            return "UNEVALUABLE -- no folds could be scored."
        auc = result["post_calibration_auc_mean"]
        lo, hi = result["post_calibration_auc_95ci"]
        ci_excludes_half = lo > 0.5 or hi < 0.5
        if auc >= AUC_GENUINE_THRESHOLD and ci_excludes_half:
            return (f"GENUINE SIGNAL -- AUC={auc:.4f} >= {AUC_GENUINE_THRESHOLD}, 95% CI [{lo:.4f},"
                    f"{hi:.4f}] excludes 0.5. A real cognitive result, free of the block confound, "
                    "underpowered but clean.")
        elif AUC_WEAK_THRESHOLD < auc < AUC_GENUINE_THRESHOLD and ci_excludes_half:
            return (f"WEAK BUT PRESENT -- AUC={auc:.4f} in ({AUC_WEAK_THRESHOLD},{AUC_GENUINE_THRESHOLD}), "
                    f"95% CI [{lo:.4f},{hi:.4f}] excludes 0.5. Report as exploratory with power caveats.")
        else:
            return (f"NO DETECTABLE CONFOUND-FREE SIGNAL -- AUC={auc:.4f}, 95% CI [{lo:.4f},{hi:.4f}] "
                    f"(contains 0.5: {not ci_excludes_half}). NOT evidence against the effect existing "
                    "-- a power statement given 0-18% minority trials per subject/condition. The "
                    "methodological paper is the paper.")

    verdicts = {name: apply_verdict(r) for name, r in
                (("within_search", result_search), ("within_memorize", result_memorize),
                 ("pooled", result_pooled))}
    for name, v in verdicts.items():
        log.info(f"  F-SME VERDICT [{name}]: {v}")

    results_payload = {
        "condition": "F-SME — subsequent memory (remembered vs. forgotten), the only within-block/"
                      "within-task/within-stimulus-set/within-session-half contrast in this dataset",
        "hyperparameters": {
            "pca_max_components": PCA_MAX_COMPONENTS, "logreg_C": LOGREG_C,
            "random_seed": RANDOM_SEED, "cov_shrinkage": COV_SHRINKAGE, "cal_fraction": CAL_FRACTION,
            "minority_min_trials": MINORITY_MIN_TRIALS,
        },
        "exclusion": exclusion, "included_subjects": included,
        "per_subject_forgotten_counts": {
            s: {"n_forgotten_search": per_subject[s]["n_forgotten_search"],
                "n_forgotten_memorize": per_subject[s]["n_forgotten_memorize"]}
            for s in sorted(per_subject.keys())
        },
        "results": {"within_search": result_search, "within_memorize": result_memorize,
                    "pooled": result_pooled},
        "decisions_md_verdicts": verdicts,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    return {
        "within_search_auc": result_search.get("post_calibration_auc_mean"),
        "within_memorize_auc": result_memorize.get("post_calibration_auc_mean"),
        "pooled_auc": result_pooled.get("post_calibration_auc_mean"),
        "verdicts": verdicts,
        "output_path": OUTPUT_JSON,
    }


@app.local_entrypoint()
def main():
    print("F-SME — subsequent memory (remembered vs. forgotten), THE LAST EXPERIMENT")
    print("The only within-block/within-task/within-stimulus-set/within-session-half contrast in "
          "ds005189 -- the block/onset/rest-break confounds cannot explain this one.")
    print("AUC is the primary metric (minority class ~0-18% per subject/condition).")
    print(f"Exclusion: subjects with < {MINORITY_MIN_TRIALS} forgotten trials in a condition are "
          "excluded from that condition entirely, computed and logged before any classification.")
    print(f"Pre-registered rule (DECISIONS.md, post_cal AUC): >= {AUC_GENUINE_THRESHOLD} with CI "
          f"excluding 0.5 -> GENUINE SIGNAL | {AUC_WEAK_THRESHOLD}-{AUC_GENUINE_THRESHOLD}, CI "
          f"excluding 0.5 -> WEAK BUT PRESENT | <= {AUC_WEAK_THRESHOLD} or CI contains 0.5 -> NO "
          "DETECTABLE SIGNAL (a power statement, not evidence against the effect).\n")
    results = run_sme_loso.remote()
    print("\nF-SME RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
