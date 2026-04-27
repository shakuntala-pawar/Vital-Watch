"""
ml_engine.py
============
Patient Deterioration Prediction System — Core ML Engine
---------------------------------------------------------
• Synthetic dataset generation calibrated to MIMIC-III / Indian ward vitals
• 10 feature families (33 features total)
• NEWS2 baseline scoring
• Manual SMOTE oversampling (no external dependency)
• 3 classifiers: Logistic Regression, Random Forest, Gradient Boosting
• Permutation importance as SHAP proxy
• 3-tier rule-based clinical alert system
• Base64-encoded chart generation for API consumption
"""

import io
import base64
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
VITALS = ["heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temperature"]

ALERT_GREEN  = 0.30   # risk below this → GREEN
ALERT_RED    = 0.80   # risk above this → RED
NEWS_AMBER   = 5      # NEWS ≥ 5 → at least AMBER
NEWS_RED     = 7      # NEWS ≥ 7 → RED override

DARK_BG   = "#0D1117"
CARD_BG   = "#111827"
ACCENT    = "#00D4FF"
SUCCESS   = "#00FF87"
WARNING   = "#FFB300"
DANGER    = "#FF4D6D"
PURPLE    = "#C084FC"
TEXT      = "#E2E8F0"
TEXT_DIM  = "#64748B"

ALERT_ACTIONS = {
    "GREEN": [
        "Continue standard 4-hourly vital observations.",
        "No immediate escalation required.",
        "Ensure next scheduled assessment is completed on time.",
    ],
    "AMBER": [
        "INCREASE observations to HOURLY immediately.",
        "Notify ward nurse-in-charge within 15 minutes.",
        "Arrange senior doctor review within 30 minutes.",
        "Reassess fluid balance and current medications.",
        "Prepare for potential escalation — have equipment ready.",
        "Document clinical state change in patient record.",
    ],
    "RED": [
        "ACTIVATE MEDICAL EMERGENCY TEAM (MET) CALL NOW.",
        "Attach patient to continuous bedside monitor immediately.",
        "Notify on-call senior resident and consultant NOW.",
        "Prepare full resuscitation equipment (crash trolley).",
        "Establish IV access if not already in place.",
        "Obtain stat ABG, CBC, metabolic panel, blood cultures.",
        "Consider ICU transfer — notify ICU registrar.",
        "Document exact time of alert and all actions taken.",
        "Do NOT leave patient unattended.",
    ],
}

# ---------------------------------------------------------------------------
# 1. DATASET GENERATION
# ---------------------------------------------------------------------------

def generate_dataset(n_patients: int = 1500, obs_per_patient: int = 12) -> pd.DataFrame:
    """Generate realistic synthetic vital-signs time-series dataset."""
    np.random.seed(42)
    records = []
    det_flags = np.random.choice([0, 1], size=n_patients, p=[0.84, 0.16])

    for pid, det in enumerate(det_flags, start=1):
        age      = int(np.random.randint(18, 90))
        sex      = np.random.choice(["M", "F"])
        comorbid = int(np.random.randint(0, 6))

        if det == 1:
            hr0, rr0, sbp0, dbp0, spo2_0, tmp0 = (
                float(np.random.normal(108, 14)),
                float(np.random.normal(24, 4)),
                float(np.random.normal(98, 18)),
                float(np.random.normal(63, 12)),
                float(np.random.normal(92, 3)),
                float(np.random.normal(38.3, 0.8)),
            )
            drift_sign = float(np.random.choice([-1.0, 1.0], p=[0.25, 0.75]))
        else:
            hr0, rr0, sbp0, dbp0, spo2_0, tmp0 = (
                float(np.random.normal(80, 11)),
                float(np.random.normal(15, 2)),
                float(np.random.normal(121, 14)),
                float(np.random.normal(78, 9)),
                float(np.random.normal(97, 1.5)),
                float(np.random.normal(37.0, 0.4)),
            )
            drift_sign = float(np.random.choice([-1.0, 1.0], p=[0.72, 0.28]))

        for t in range(obs_per_patient):
            d = t * 0.3 * drift_sign if det == 1 else t * 0.04 * drift_sign

            hr   = float(np.clip(hr0   + d * 1.5  + np.random.normal(0, 5),    30, 200))
            rr   = float(np.clip(rr0   + d * 0.4  + np.random.normal(0, 2),     5,  40))
            sbp  = float(np.clip(sbp0  - d * 1.2  + np.random.normal(0, 8),    60, 220))
            dbp  = float(np.clip(dbp0  - d * 0.8  + np.random.normal(0, 5),    30, 130))
            spo2 = float(np.clip(spo2_0 - d * 0.3 + np.random.normal(0, 1),   70, 100))
            tmp  = float(np.clip(tmp0  + d * 0.05 + np.random.normal(0, 0.3), 34,  42))

            spo2_val = float("nan") if np.random.random() < 0.04 else round(spo2, 1)
            tmp_val  = float("nan") if np.random.random() < 0.06 else round(tmp, 2)

            records.append({
                "patient_id":        pid,
                "age":               age,
                "sex":               sex,
                "comorbidity_score": comorbid,
                "time_step":         t,
                "heart_rate":        round(hr, 1),
                "resp_rate":         round(rr, 1),
                "sbp":               round(sbp, 1),
                "dbp":               round(dbp, 1),
                "spo2":              spo2_val,
                "temperature":       tmp_val,
                "deterioration":     int(det),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. NEWS2 SCORE
# ---------------------------------------------------------------------------

def compute_news(row: dict) -> int:
    """Compute NEWS2 score from a dict of vital values."""
    score = 0

    rr = row.get("resp_rate")
    if rr is not None and not (isinstance(rr, float) and np.isnan(rr)):
        rr = float(rr)
        if   rr <= 8:  score += 3
        elif rr <= 11: score += 1
        elif rr <= 20: score += 0
        elif rr <= 24: score += 2
        else:          score += 3

    spo2 = row.get("spo2")
    if spo2 is not None and not (isinstance(spo2, float) and np.isnan(spo2)):
        spo2 = float(spo2)
        if   spo2 <= 91: score += 3
        elif spo2 <= 93: score += 2
        elif spo2 <= 95: score += 1

    sbp = row.get("sbp")
    if sbp is not None and not (isinstance(sbp, float) and np.isnan(sbp)):
        sbp = float(sbp)
        if   sbp <= 90:  score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1
        elif sbp <= 219: score += 0
        else:            score += 3

    hr = row.get("heart_rate")
    if hr is not None and not (isinstance(hr, float) and np.isnan(hr)):
        hr = float(hr)
        if   hr <= 40:  score += 3
        elif hr <= 50:  score += 1
        elif hr <= 90:  score += 0
        elif hr <= 110: score += 1
        elif hr <= 130: score += 2
        else:           score += 3

    tmp = row.get("temperature")
    if tmp is not None and not (isinstance(tmp, float) and np.isnan(tmp)):
        tmp = float(tmp)
        if   tmp <= 35.0: score += 3
        elif tmp <= 36.0: score += 1
        elif tmp <= 38.0: score += 0
        elif tmp <= 39.0: score += 1
        else:             score += 2

    return score


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-patient time-series into a flat feature vector (33 features)."""
    records = []
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("time_step").reset_index(drop=True)
        lat = grp.iloc[-1]
        f: dict = {"patient_id": int(pid)}

        # F7: static demographics
        f["age"]               = float(lat["age"])
        f["comorbidity_score"] = float(lat["comorbidity_score"])
        f["sex_male"]          = 1.0 if lat["sex"] == "M" else 0.0

        # F1: latest vital values
        for v in VITALS:
            series = grp[v].dropna()
            f[f"{v}_last"] = float(series.iloc[-1]) if len(series) > 0 else 0.0

        # F2: trend slopes (linear regression over all obs)
        t_arr = grp["time_step"].values.astype(float)
        for v in VITALS:
            vals = grp[v].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() >= 3:
                slope = float(np.polyfit(t_arr[mask], vals[mask], 1)[0])
            else:
                slope = 0.0
            f[f"{v}_slope"] = slope

        # F3: variability (std)
        for v in VITALS:
            std = grp[v].std(skipna=True)
            f[f"{v}_std"] = float(std) if not np.isnan(std) else 0.0

        # F4: rolling range (last 4 observations)
        last4 = grp.tail(4)
        for v in VITALS:
            hi = last4[v].max()
            lo = last4[v].min()
            rng = hi - lo
            f[f"{v}_range4"] = float(rng) if not np.isnan(rng) else 0.0

        # F5: NEWS2 score from latest reading
        f["news_score"] = float(compute_news(dict(lat)))

        # F6: missingness fraction
        f["spo2_missing_pct"]       = float(grp["spo2"].isna().mean())
        f["temperature_missing_pct"] = float(grp["temperature"].isna().mean())

        # F8: shock index  HR / SBP
        hr_l  = f.get("heart_rate_last", 0.0)
        sbp_l = f.get("sbp_last", 1.0)
        f["shock_index"] = hr_l / sbp_l if sbp_l > 0 else 0.0

        # F9: pulse pressure  SBP - DBP
        dbp_l = f.get("dbp_last", 0.0)
        f["pulse_pressure"] = sbp_l - dbp_l

        # F10: HR / RR ratio
        rr_l = f.get("resp_rate_last", 1.0)
        f["hr_rr_ratio"] = hr_l / rr_l if rr_l > 0 else 0.0

        f["deterioration"] = int(lat["deterioration"])
        records.append(f)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. NEWS BASELINE EVALUATOR
# ---------------------------------------------------------------------------

def news_baseline(feat_df: pd.DataFrame, threshold: int = 5) -> dict:
    y      = feat_df["deterioration"].values
    scores = feat_df["news_score"].values
    max_s  = float(scores.max()) if scores.max() > 0 else 1.0
    yprob  = scores / max_s
    ypred  = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, ypred).ravel()
    return {
        "auroc":       float(roc_auc_score(y, yprob)),
        "auprc":       float(average_precision_score(y, yprob)),
        "f1":          float(f1_score(y, ypred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "y_prob":      yprob,
        "y_pred":      ypred,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# 5. SMOTE (manual, no imbalanced-learn dependency)
# ---------------------------------------------------------------------------

def smote_oversample(X: np.ndarray, y: np.ndarray, ratio: float = 0.8):
    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    n_maj, n_min = len(maj_idx), len(min_idx)
    n_need = int(n_maj * ratio) - n_min
    if n_need <= 0:
        return X, y

    X_min = X[min_idx]
    k     = min(5, n_min - 1)
    nn    = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)

    rng = np.random.RandomState(42)
    synthetic = []
    for _ in range(n_need):
        base = rng.randint(0, n_min)
        nbr  = indices[base, rng.randint(1, k + 1)]
        alpha = rng.random()
        synthetic.append(X_min[base] + alpha * (X_min[nbr] - X_min[base]))

    X_syn = np.vstack(synthetic)
    y_syn = np.ones(len(synthetic), dtype=int)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# ---------------------------------------------------------------------------
# 6. MODEL TRAINING
# ---------------------------------------------------------------------------

def train_models(X_tr, y_tr, X_te, y_te, feat_names: list) -> dict:
    model_defs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1200, C=0.5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        ),
    }

    results = {}
    for name, mdl in model_defs.items():
        mdl.fit(X_tr, y_tr)
        yp    = mdl.predict(X_te)
        yprob = mdl.predict_proba(X_te)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
        results[name] = {
            "model":       mdl,
            "y_pred":      yp,
            "y_prob":      yprob,
            "auroc":       float(roc_auc_score(y_te, yprob)),
            "auprc":       float(average_precision_score(y_te, yprob)),
            "f1":          float(f1_score(y_te, yp)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "ppv":         float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "npv":         float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    return results


# ---------------------------------------------------------------------------
# 7. PERMUTATION IMPORTANCE  (SHAP proxy)
# ---------------------------------------------------------------------------

def compute_importance(model, X_te, y_te, feat_names, n_repeats=20) -> pd.DataFrame:
    res = permutation_importance(
        model, X_te, y_te,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature":    feat_names,
        "importance": res.importances_mean,
        "std":        res.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 8. ALERT SYSTEM
# ---------------------------------------------------------------------------

def run_alert(risk_score: float, news_score: int, patient_id: str = "PT-001") -> dict:
    risk = float(risk_score)
    news = int(news_score)

    if news >= NEWS_RED:
        level   = "RED"
        trigger = f"NEWS score {news} ≥ {NEWS_RED} (hard clinical override)"
    elif risk >= ALERT_RED:
        level   = "RED"
        trigger = f"ML risk score {risk:.3f} ≥ {ALERT_RED} (RED threshold)"
    elif risk >= ALERT_GREEN or news >= NEWS_AMBER:
        level   = "AMBER"
        trigger = (
            f"ML risk score {risk:.3f} ≥ {ALERT_GREEN} (AMBER threshold)"
            if risk >= ALERT_GREEN
            else f"NEWS score {news} ≥ {NEWS_AMBER} (AMBER threshold)"
        )
    else:
        level   = "GREEN"
        trigger = f"ML risk score {risk:.3f} — within safe range"

    return {
        "patient_id":  patient_id,
        "risk_score":  round(risk, 4),
        "news_score":  news,
        "alert_level": level,
        "trigger":     trigger,
        "actions":     ALERT_ACTIONS[level],
        "thresholds":  {
            "green_max":  ALERT_GREEN,
            "red_min":    ALERT_RED,
            "news_amber": NEWS_AMBER,
            "news_red":   NEWS_RED,
        },
    }


# ---------------------------------------------------------------------------
# 9. CHART HELPERS
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def _dark_axes(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="#999999")
    for sp in ax.spines.values():
        sp.set_color("#1E293B")
    ax.grid(True, color="#1E293B", linewidth=0.6)


def chart_roc(results: dict, news_y_prob, y_te) -> str:
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=DARK_BG)
    _dark_axes(ax)
    cols = [ACCENT, SUCCESS, WARNING, DANGER]
    for (name, res), col in zip(results.items(), cols):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax.plot(fpr, tpr, color=col, lw=2.2,
                label=f"{name} (AUC={res['auroc']:.3f})")
    fpr_n, tpr_n, _ = roc_curve(y_te, news_y_prob)
    ax.plot(fpr_n, tpr_n, color="#888888", lw=1.8, ls="--",
            label=f"NEWS baseline")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("1 - Specificity", color=TEXT, fontsize=11)
    ax.set_ylabel("Sensitivity", color=TEXT, fontsize=11)
    ax.set_title("ROC Curves — All Models vs NEWS Baseline",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor=TEXT,
              edgecolor="#1E293B")
    return _fig_to_b64(fig)


def chart_pr(results: dict, news_y_prob, y_te, prevalence: float) -> str:
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=DARK_BG)
    _dark_axes(ax)
    cols = [ACCENT, SUCCESS, WARNING, DANGER]
    for (name, res), col in zip(results.items(), cols):
        prec, rec, _ = precision_recall_curve(y_te, res["y_prob"])
        ax.plot(rec, prec, color=col, lw=2.2,
                label=f"{name} (AP={res['auprc']:.3f})")
    pn, rn, _ = precision_recall_curve(y_te, news_y_prob)
    ax.plot(rn, pn, color="#888888", lw=1.8, ls="--", label="NEWS baseline")
    ax.axhline(prevalence, color=DANGER, ls=":", lw=1,
               label=f"Prevalence ({prevalence:.2f})")
    ax.set_xlabel("Recall", color=TEXT, fontsize=11)
    ax.set_ylabel("Precision", color=TEXT, fontsize=11)
    ax.set_title("Precision-Recall Curves\n(Critical for Imbalanced Classes)",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor=TEXT,
              edgecolor="#1E293B")
    return _fig_to_b64(fig)


def chart_confusion(y_te, y_pred, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    cm   = confusion_matrix(y_te, y_pred)
    cmap = LinearSegmentedColormap.from_list("", [DARK_BG, ACCENT])
    ax.imshow(cm, cmap=cmap, aspect="auto")
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=22, fontweight="bold",
                color="#FFF" if cm[i, j] > cm.max() / 2 else ACCENT)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Stable", "Pred: Deteriorating"],
                       color="#CCC", fontsize=10)
    ax.set_yticklabels(["True: Stable", "True: Deteriorating"],
                       color="#CCC", fontsize=10)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 color=TEXT, fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_color("#1E293B")
    return _fig_to_b64(fig)


def chart_importance(imp_df: pd.DataFrame) -> str:
    top = imp_df.head(15)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK_BG)
    _dark_axes(ax)
    bar_cols = []
    for feat in top["feature"]:
        if "slope" in feat:                        bar_cols.append(DANGER)
        elif "std" in feat or "range" in feat:     bar_cols.append(WARNING)
        elif "news" in feat:                        bar_cols.append(SUCCESS)
        elif feat in {"shock_index","pulse_pressure","hr_rr_ratio"}: bar_cols.append(PURPLE)
        else:                                       bar_cols.append(ACCENT)
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            xerr=top["std"][::-1], color=bar_cols[::-1],
            edgecolor=DARK_BG, capsize=3, alpha=0.9, height=0.65)
    ax.set_title("Top-15 Feature Importances (Permutation / SHAP Proxy)\n"
                 "🔴 Trend  🟡 Variability  🟢 NEWS  🟣 Derived  🔵 Raw",
                 color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean AUROC drop when feature is shuffled", color="#CCC")
    ax.tick_params(axis="y", labelsize=9, colors="#CCC")
    ax.tick_params(axis="x", colors="#999")
    return _fig_to_b64(fig)


def chart_risk_dist(y_prob, y_te) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=DARK_BG)
    _dark_axes(ax)
    ax.hist(y_prob[y_te == 0], bins=30, alpha=0.65, color=ACCENT,
            label="Stable", density=True)
    ax.hist(y_prob[y_te == 1], bins=30, alpha=0.65, color=DANGER,
            label="Deteriorating", density=True)
    ax.axvline(ALERT_GREEN, color=SUCCESS, lw=2, ls="--",
               label=f"GREEN→AMBER ({ALERT_GREEN})")
    ax.axvline(ALERT_RED,   color=DANGER,  lw=2, ls=":",
               label=f"AMBER→RED ({ALERT_RED})")
    ax.set_xlabel("Predicted Risk Score", color=TEXT, fontsize=11)
    ax.set_ylabel("Density", color=TEXT, fontsize=11)
    ax.set_title("Risk Score Distribution with Alert Thresholds",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor=TEXT,
              edgecolor="#1E293B")
    return _fig_to_b64(fig)


def chart_sens_spec(results: dict, news_res: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK_BG)
    _dark_axes(ax)
    names = list(results.keys()) + ["NEWS Baseline"]
    sens  = [results[n]["sensitivity"] for n in results] + [news_res["sensitivity"]]
    spec  = [results[n]["specificity"] for n in results] + [news_res["specificity"]]
    x = np.arange(len(names))
    w = 0.35
    bars_s  = ax.bar(x - w / 2, sens, w, color=DANGER,  alpha=0.9, label="Sensitivity")
    bars_sp = ax.bar(x + w / 2, spec, w, color=ACCENT, alpha=0.9, label="Specificity")
    ax.axhline(0.80, color=WARNING, ls="--", lw=1.5, label="Clinical target (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", color="#CCC", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Sensitivity vs Specificity — Models vs NEWS",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor=TEXT,
              edgecolor="#1E293B")
    for i, (s, sp) in enumerate(zip(sens, spec)):
        ax.text(i - w / 2, s + 0.01, f"{s:.2f}", ha="center",
                fontsize=8, color=DANGER, fontweight="bold")
        ax.text(i + w / 2, sp + 0.01, f"{sp:.2f}", ha="center",
                fontsize=8, color=ACCENT, fontweight="bold")
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# 10. PATIENT PREDICTION  (single row inference)
# ---------------------------------------------------------------------------

def predict_patient(data: dict, state: dict) -> dict:
    if not state:
        return {"error": "Model not trained yet."}

    scaler    = state["scaler"]
    model     = state["best_model"]
    feat_cols = state["feat_cols"]
    col_meds  = state["col_medians"]
    imp_df    = state["imp_df"]

    raw = {
        "heart_rate":  float(data.get("heart_rate",  80)),
        "resp_rate":   float(data.get("resp_rate",   16)),
        "sbp":         float(data.get("sbp",        120)),
        "dbp":         float(data.get("dbp",         78)),
        "spo2":        float(data.get("spo2",        97)),
        "temperature": float(data.get("temperature", 37.0)),
    }

    news = compute_news(raw)
    shock  = raw["heart_rate"] / raw["sbp"]   if raw["sbp"]      > 0 else 0.0
    pp     = raw["sbp"] - raw["dbp"]
    hr_rr  = raw["heart_rate"] / raw["resp_rate"] if raw["resp_rate"] > 0 else 0.0

    feat_map: dict = {}
    for v in VITALS:
        feat_map[f"{v}_last"]   = raw.get(v, 0.0)
        feat_map[f"{v}_slope"]  = float(data.get(f"{v}_slope",  0.0))
        feat_map[f"{v}_std"]    = float(data.get(f"{v}_std",    0.0))
        feat_map[f"{v}_range4"] = float(data.get(f"{v}_range4", 0.0))

    feat_map["news_score"]              = float(news)
    feat_map["spo2_missing_pct"]        = 0.0
    feat_map["temperature_missing_pct"] = 0.0
    feat_map["shock_index"]             = shock
    feat_map["pulse_pressure"]          = pp
    feat_map["hr_rr_ratio"]             = hr_rr
    feat_map["age"]                     = float(data.get("age", 50))
    feat_map["comorbidity_score"]       = float(data.get("comorbidity_score", 1))
    feat_map["sex_male"]                = 1.0 if str(data.get("sex","M")).upper() == "M" else 0.0

    x_raw = np.array([[feat_map.get(c, 0.0) for c in feat_cols]], dtype=float)
    for j in range(x_raw.shape[1]):
        if np.isnan(x_raw[0, j]):
            x_raw[0, j] = col_meds[j]

    x_scaled = scaler.transform(x_raw)
    risk      = float(model.predict_proba(x_scaled)[0, 1])
    alert     = run_alert(risk, news, str(data.get("patient_id", "PT-001")))

    # Feature contributions  (signed importance × standardised deviation)
    mu  = scaler.mean_
    sig = scaler.scale_
    imp_lookup = dict(zip(imp_df["feature"], imp_df["importance"]))
    contribs = []
    for j, col in enumerate(feat_cols):
        dev  = (x_raw[0, j] - mu[j]) / (sig[j] + 1e-9)
        imp_v = imp_lookup.get(col, 0.0)
        contribs.append({
            "feature":      col,
            "value":        round(float(x_raw[0, j]), 3),
            "contribution": round(float(dev * imp_v), 6),
        })
    contribs.sort(key=lambda r: abs(r["contribution"]), reverse=True)

    return {
        "risk_score":            round(risk, 4),
        "news_score":            news,
        "alert":                 alert,
        "feature_contributions": contribs[:12],
        "model_used":            state["best_name"],
        "model_auroc":           round(state["results"][state["best_name"]]["auroc"], 4),
    }


# ---------------------------------------------------------------------------
# 11. BATCH CSV PREDICTION
# ---------------------------------------------------------------------------

def batch_predict_csv(csv_text: str, state: dict) -> list:
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        return [{"error": f"CSV parse error: {exc}"}]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required   = {"heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temperature"}
    missing    = required - set(df.columns)
    if missing:
        return [{"error": f"Missing columns: {missing}"}]

    defaults = {
        "heart_rate": 80, "resp_rate": 16, "sbp": 120, "dbp": 78,
        "spo2": 97, "temperature": 37.0, "age": 50, "comorbidity_score": 1,
    }

    out = []
    for idx, row in df.iterrows():
        data = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                data[k] = defaults.get(k, 0)
            else:
                data[k] = v
        result = predict_patient(data, state)
        result["patient_id"] = str(data.get("patient_id", f"Row-{idx + 1}"))
        out.append(result)

    return out


# ---------------------------------------------------------------------------
# 12. MASTER TRAINING PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """Execute the full ML pipeline. Returns state dict."""

    # ── Data
    df      = generate_dataset(n_patients=1500, obs_per_patient=12)
    feat_df = engineer_features(df)

    # ── Prepare arrays
    drop_cols = ["patient_id", "deterioration"]
    feat_cols = [c for c in feat_df.columns if c not in drop_cols]
    X         = feat_df[feat_cols].values.astype(float)
    y         = feat_df["deterioration"].values.astype(int)
    news_all  = feat_df["news_score"].values.astype(float)

    # Impute NaN with training-set median (fit on all; split next)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_meds[j]

    # ── Train / test split (stratified), keep index alignment for NEWS
    all_idx          = np.arange(len(y))
    idx_tr, idx_te   = train_test_split(all_idx, test_size=0.2,
                                         stratify=y, random_state=42)
    X_tr, X_te       = X[idx_tr], X[idx_te]
    y_tr, y_te       = y[idx_tr], y[idx_te]
    news_te           = news_all[idx_te]          # aligned test NEWS scores

    # ── Scale
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # ── SMOTE on training only
    X_tr_b, y_tr_b = smote_oversample(X_tr_s, y_tr, ratio=0.8)
    smote_before   = {"minority": int((y_tr == 1).sum()), "majority": int((y_tr == 0).sum())}
    smote_after    = {"minority": int((y_tr_b == 1).sum()), "majority": int((y_tr_b == 0).sum())}

    # ── Train models
    results = train_models(X_tr_b, y_tr_b, X_te_s, y_te, feat_cols)

    # ── Best model
    best_name = max(results, key=lambda n: results[n]["auroc"])
    best_mdl  = results[best_name]["model"]

    # ── NEWS baseline (full dataset for summary; test-aligned for plots)
    nr          = news_baseline(feat_df, threshold=NEWS_AMBER)
    max_news    = float(news_te.max()) if news_te.max() > 0 else 1.0
    news_te_prob = news_te / max_news          # test-set NEWS probabilities

    # ── Feature importance
    imp_df = compute_importance(best_mdl, X_te_s, y_te, feat_cols, n_repeats=20)

    prevalence = float(feat_df["deterioration"].mean())

    # ── Charts
    charts = {
        "roc":        chart_roc(results, news_te_prob, y_te),
        "pr":         chart_pr(results, news_te_prob, y_te, prevalence),
        "confusion":  chart_confusion(y_te, results[best_name]["y_pred"], best_name),
        "importance": chart_importance(imp_df),
        "risk_dist":  chart_risk_dist(results[best_name]["y_prob"], y_te),
        "sens_spec":  chart_sens_spec(results, nr),
    }

    # ── Serialisable metrics
    metrics = {}
    for name, r in results.items():
        metrics[name] = {
            "auroc":       round(r["auroc"], 4),
            "auprc":       round(r["auprc"], 4),
            "f1":          round(r["f1"], 4),
            "sensitivity": round(r["sensitivity"], 4),
            "specificity": round(r["specificity"], 4),
            "ppv":         round(r["ppv"], 4),
            "npv":         round(r["npv"], 4),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        }
    metrics["NEWS Baseline"] = {
        "auroc":       round(nr["auroc"], 4),
        "auprc":       round(nr["auprc"], 4),
        "f1":          round(nr["f1"], 4),
        "sensitivity": round(nr["sensitivity"], 4),
        "specificity": round(nr["specificity"], 4),
        "ppv": None, "npv": None,
        "tp": nr["tp"], "fp": nr["fp"], "fn": nr["fn"], "tn": nr["tn"],
    }

    top_features = [
        {
            "feature":    row["feature"],
            "importance": round(float(row["importance"]), 6),
            "std":        round(float(row["std"]), 6),
        }
        for _, row in imp_df.head(12).iterrows()
    ]

    return {
        # Inference artefacts
        "scaler":      scaler,
        "best_model":  best_mdl,
        "best_name":   best_name,
        "feat_cols":   feat_cols,
        "col_medians": col_meds,
        # Analytics
        "results":     results,
        "news_res":    nr,
        "imp_df":      imp_df,
        "feat_df":     feat_df,
        "charts":      charts,
        "metrics":     metrics,
        "top_features": top_features,
        # Summary stats
        "prevalence":   prevalence,
        "n_patients":   int(len(feat_df)),
        "n_features":   int(len(feat_cols)),
        "smote_before": smote_before,
        "smote_after":  smote_after,
    }
