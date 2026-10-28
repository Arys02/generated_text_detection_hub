import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, det_curve, precision_recall_fscore_support, accuracy_score, auc


def get_tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, fpr_target: float = 0.05) -> float:
    """
    Calcule le True Positive Rate (Recall) pour un False Positive Rate fixe.
    Crucial pour les applications réelles où le coût d'un faux positif est élevé.
    """
    fpr, tpr, thresholds = get_roc_metrics(y_true, y_scores)

    return float(np.interp(fpr_target, fpr, tpr))

def get_roc_metrics(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true,  y_score)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve(y_true, -y_score)
        roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def _is_inverted(y_true: np.ndarray, y_scores: np.ndarray) -> bool:
    return roc_auc_score(y_true, y_scores) < 0.5

# utile sur le TT
def get_standards_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5):
    print(_is_inverted(y_true, y_scores), threshold)
    if _is_inverted(y_true, y_scores):
        y_pred = (y_scores < threshold).astype(int)
    else:
        y_pred = (y_scores >= threshold).astype(int)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1_score, accuracy


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold) -> dict:
    """Retourne un dictionnaire standardisé de métriques."""
    try:
        _, _, auroc = get_roc_metrics(y_true, y_scores)
        tpr_5 = get_tpr_at_fpr(y_true, y_scores, fpr_target=0.05)
        tpr_1 = get_tpr_at_fpr(y_true, y_scores, fpr_target=0.01)
        precision, recall, f1_score, accuracy = get_standards_metrics(y_true, y_scores, threshold)

        return {
            "auroc": auroc,
            "tpr_5_fpr": tpr_5,
            "tpr_1_fpr": tpr_1,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "y_true": y_true.tolist(),
            "y_scores": y_scores.tolist(),
        }
    except ValueError:
        # Gère le cas où le batch ne contient qu'une seule classe (crash sklearn)
        return {"auroc": 0.0, "tpr@5%fpr": 0.0}

def roc_at_fpr_budgets(label: np.ndarray, target: np.ndarray, fpr_budgets=[0.01, 0.05, 0.10], inverted=False):
    s = -np.asarray(target, float) if inverted else np.asarray(target, float)
    fpr, tpr, thr = roc_curve(label, s)
    roc_auc = auc(fpr, tpr)

    fpr_budgets = np.atleast_1d(fpr_budgets)
    cmap = plt.cm.autumn(np.linspace(0, 0.85, len(fpr_budgets)))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="tab:blue", lw=1.5, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)

    results = []
    for budget, color in zip(fpr_budgets, cmap):
        admissible = np.where(fpr <= budget)[0]
        if admissible.size == 0:
            print(f"[skip] Aucun point sous FPR={budget:.2%} "
                  f"(FPR min = {fpr[fpr > 0].min():.4f})")
            continue
        idx = admissible[-1]  # tpr monotone croissant => dernier indice = meilleur TPR admissible
        thr_native = -thr[idx] if inverted else thr[idx]
        fpr_op, tpr_op = fpr[idx], tpr[idx]

        plt.axvline(budget, color=color, ls=":", alpha=0.7)
        lbl = f"TPR@{budget:.0%}FPR = {tpr_op:.3f}"
        lbl += "" if np.isinf(thr_native) else f" (thr={thr_native:.4f})"
        plt.scatter([fpr_op], [tpr_op], color=color, zorder=5, label=lbl)

        results.append((budget, roc_auc, fpr_op, tpr_op, thr_native))
        print(f"FPR={budget:.0%} | TPR={tpr_op:.4f} | FPR_eff={fpr_op:.4f} | thr={thr_native:.6f}")

    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curve"); plt.legend(loc="lower right")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    print(f"\nAUROC : {roc_auc:.4f}")
    return roc_auc, results