"""Stage 5: Train and cross-validate ChordNet across all 9 selected chords.

Extracts exact runtime features (64 cols via geometry_features.frame_features) from
all 785 frames in artifacts/own_videos/*/frames/*.jpg into features_abs_64.csv,
then runs strict leave-one-take-out cross-validation so training and live inference
use the exact same absolute fret/string coordinate space.

Usage:
    python scripts/train_chord_model.py
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry_features import HandFinder, NeckFinder, feature_vector, frame_features

CHORDS = ["A", "Am", "Bm", "C", "D", "Dm", "E", "F", "G"]
SMOOTH_K = 5
SEED = 42


def extract_abs_features(out_csv: Path) -> pd.DataFrame:
    neck_path = ROOT / "artifacts" / "models_local" / "neck_v2.pt"
    hand_path = ROOT / "artifacts" / "models_local" / "hand_landmarker.task"
    neck, hand = NeckFinder(neck_path), HandFinder(hand_path)
    names = feature_vector(np.zeros(21), np.zeros(21))[1]

    rows = []
    frames = sorted((ROOT / "artifacts" / "own_videos").glob("*/frames/*.jpg"))
    print(f"Extracting exact runtime features from {len(frames)} frames...")
    for idx, fpath in enumerate(frames):
        chord = fpath.parents[1].name
        parts = fpath.stem.split("_")
        take = parts[1] if len(parts) >= 2 else "take1"
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            continue
        r = frame_features(bgr, neck, hand)
        row = {
            "id": fpath.stem,
            "chord": chord,
            "take": take,
            "neck_valid": r["neck_valid"],
            "hand_found": r["hand_found"],
            "inside": r["inside"],
            "usable": r["feat"] is not None,
        }
        vals = r["feat"] if r["feat"] is not None else [np.nan] * len(names)
        for n, x in zip(names, vals):
            row[n] = float(x)
        rows.append(row)
        if (idx + 1) % 200 == 0:
            print(f"  processed {idx + 1}/{len(frames)} frames...", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows, {int(df['usable'].sum())} usable)")
    return df


def knn_distance(X, ref, mean, std, k=5):
    A = (np.asarray(ref, np.float64) - mean) / std
    B = (np.atleast_2d(np.asarray(X, np.float64)) - mean) / std
    D = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(2))
    D.sort(1)
    return D[:, :k].mean(1)


def augment_rows(X, y, times=3, sigma=0.03, seed=SEED):
    r = np.random.default_rng(seed)
    sigma = np.broadcast_to(np.asarray(sigma, float), (X.shape[1],))
    return np.vstack([X] + [X + r.normal(0, sigma, X.shape) for _ in range(times)]), np.concatenate([y] * (times + 1))


def smooth_accuracy(P, y, groups, k=SMOOTH_K, classes=CHORDS):
    ok = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in range(len(idx)):
            win = P[idx[max(0, j - k + 1) : j + 1]].mean(0)
            ok.append(classes[int(win.argmax())] == y[idx[j]])
    return float(np.mean(ok))


def main():
    print("=" * 70)
    print("STAGE 5: Training & Evaluating 9-Chord Recognition Pipeline (Absolute Runtime Space)")
    print("=" * 70)

    csv_path = ROOT / "artifacts" / "own_videos" / "features_abs_64.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded cached {csv_path.name} ({len(df)} rows)")
    else:
        df = extract_abs_features(csv_path)

    U = df[(df["usable"] == True) & (df["chord"].isin(CHORDS))].copy().reset_index(drop=True)
    print(f"Usable fretting hand frames: {len(U)} (Chance = {100 / len(CHORDS):.1f}%)")

    tab_cols = [f"str{i}" for i in range(4)] + [f"fr{i}" for i in range(4)]
    span_cols = ["d_fr_01", "d_fr_12", "d_fr_02", "d_fr_23", "d_str_01", "d_str_12", "d_str_02", "span_fr", "span_str"]

    feature_sets = {
        "tab_only (8 numbers)": tab_cols,
        "tab + relative spans (17 numbers)": tab_cols + span_cols,
    }

    models = {
        "RandomForest (150 trees)": lambda: RandomForestClassifier(150, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=120, min_samples_leaf=4, random_state=SEED),
        "Ensemble (RF + HGB)": lambda: VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(150, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(max_iter=120, min_samples_leaf=4, random_state=SEED)),
            ],
            voting="soft",
            n_jobs=-1,
        ),
    }

    classes = sorted(U["chord"].unique())
    takes = sorted(U["take"].unique())
    groups = (U["chord"] + "_" + U["take"]).values
    y = U["chord"].values

    best_score = -1.0
    best_config = None
    best_P = None
    best_model_obj = None
    results = []

    for f_name, cols in feature_sets.items():
        X = U[cols].values.astype(np.float32)
        for m_name, model_fn in models.items():
            P = np.zeros((len(U), len(classes)))
            for tk in takes:
                tr_mask = U["take"] != tk
                te_mask = U["take"] == tk
                if not te_mask.any():
                    continue
                X_tr, y_tr = X[tr_mask], y[tr_mask]
                X_te = X[te_mask]
                sigmas = [0.20 if "str" in c or "fr" in c else 0.04 for c in cols]
                X_tr_aug, y_tr_aug = augment_rows(X_tr, y_tr, times=2, sigma=sigmas)
                clf = model_fn()
                clf.fit(X_tr_aug, y_tr_aug)
                pr = clf.predict_proba(X_te)
                for k_i, cls in enumerate(clf.classes_):
                    P[np.where(te_mask)[0], classes.index(cls)] = pr[:, k_i]

            acc_raw = float(np.mean(np.array(classes)[P.argmax(1)] == y))
            acc_smooth = smooth_accuracy(P, y, groups, k=SMOOTH_K, classes=classes)
            combined_score = 0.4 * acc_raw + 0.6 * acc_smooth
            results.append({
                "Features": f_name,
                "Model": m_name,
                "Per-Frame": acc_raw,
                "5-Frame Averaged": acc_smooth,
                "Combined Score": combined_score,
            })
            print(f"[{f_name} | {m_name}] -> Raw: {acc_raw*100:.1f}% | 5-Frame Smoothed: {acc_smooth*100:.1f}%")
            if combined_score > best_score:
                best_score = combined_score
                best_config = (f_name, m_name, cols)
                best_P = P
                best_model_obj = model_fn

    res_df = pd.DataFrame(results).sort_values("Combined Score", ascending=False)
    f_name, m_name, cols = best_config
    print(f"\nWINNER: {f_name} with {m_name}")

    print("\n--- Per-Chord Held-Out Recall ---")
    for ci in classes:
        mask_c = y == ci
        raw_c = np.mean(np.array(classes)[best_P.argmax(1)][mask_c] == ci)
        smoothed_hits = []
        for g in np.unique(groups):
            idx = np.where((groups == g) & (y == ci))[0]
            if len(idx) == 0:
                continue
            all_g_idx = np.where(groups == g)[0]
            for j in idx:
                rel_j = np.where(all_g_idx == j)[0][0]
                win = best_P[all_g_idx[max(0, rel_j - SMOOTH_K + 1) : rel_j + 1]].mean(0)
                smoothed_hits.append(classes[int(win.argmax())] == ci)
        sm_c = np.mean(smoothed_hits) if smoothed_hits else 0.0
        print(f"Chord {ci:2s} (n={mask_c.sum():2d}): Raw Recall = {raw_c*100:5.1f}% | Smoothed Recall = {sm_c*100:5.1f}%")

    X_all = U[cols].values.astype(np.float32)
    sigmas = [0.20 if "str" in c or "fr" in c else 0.04 for c in cols]
    X_all_aug, y_all_aug = augment_rows(X_all, y, times=2, sigma=sigmas)
    final_clf = best_model_obj()
    final_clf.fit(X_all_aug, y_all_aug)

    tk = U["take"].values
    din = np.zeros(len(U))
    for t_ in np.unique(tk):
        te_ = tk == t_
        tr_ = ~te_
        din[te_] = knn_distance(X_all[te_], X_all[tr_], X_all[tr_].mean(0), X_all[tr_].std(0) + 1e-6, 5)
    novelty_thr = float(np.percentile(din, 97))
    print(f"\nNovelty filter threshold (97th percentile): {novelty_thr:.3f}")
    print(f"fr0..fr3 mean in runtime space: {X_all[:, 4:8].mean(0).round(2)}")

    meta = {
        "features": f_name,
        "model": m_name,
        "columns": cols,
        "classes": classes,
        "smooth_k": SMOOTH_K,
        "cv_per_frame": float(res_df.iloc[0]["Per-Frame"]),
        "cv_averaged": float(res_df.iloc[0]["5-Frame Averaged"]),
        "novelty_keep": 0.97,
        "novelty_thr": novelty_thr,
    }
    bundle = {
        "model": final_clf,
        "meta": meta,
        "novelty": {"ref": X_all, "mean": X_all.mean(0), "std": X_all.std(0) + 1e-6, "k": 5, "thr": novelty_thr},
    }

    out_clf = ROOT / "artifacts" / "models_local" / "chord_clf.joblib"
    out_meta = ROOT / "artifacts" / "models_local" / "chord_meta.json"
    joblib.dump(bundle, out_clf)
    out_meta.write_text(json.dumps(meta, indent=1))
    print(f"Exported aligned 9-chord model to {out_clf} and {out_meta}")


if __name__ == "__main__":
    main()
