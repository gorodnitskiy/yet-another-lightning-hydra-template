from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def accuracy(
    targets: List[int], preds: List[int], verbose: bool = True
) -> float:
    acc = accuracy_score(targets, preds)
    if verbose:
        print(f"Full: accuracy={acc:.5f}")
    for target_i in np.unique(targets):
        curr_targets = []
        curr_preds = []
        for i in range(len(targets)):
            if targets[i] == target_i:
                curr_targets.append(targets[i])
                curr_preds.append(preds[i])
        acc_i = accuracy_score(curr_targets, curr_preds)
        if verbose:
            print(f"class {target_i}: accuracy={acc_i:.5f}")

    return acc


def auroc(
    targets: List[int],
    probs: List[float],
    plot: bool = False,
    path: str = "",
    verbose: bool = True,
) -> float:
    """Calculate and draw Area Under Receiver Operating Characteristic Curve."""
    roc_auc = roc_auc_score(targets, probs)
    if verbose:
        print(f"AUROC={roc_auc:.5f}")
    # draw plot
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, marker=".", label=f"AUROC={roc_auc:.5f}")
    plt.xlabel("False Positive Rate")
    plt.xlim(0, 1)
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.legend()

    if path:
        plt.savefig(path)

    if plot:
        plt.show()
    plt.close()

    return roc_auc


def auprc(
    targets: List[int],
    probs: List[float],
    plot: bool = False,
    path: str = "",
    verbose: bool = True,
) -> float:
    """Calculate and draw Area Under Precision-Recall Curve."""
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)
    if verbose:
        print(f"AUPRC={pr_auc:.5f}")
    # draw plot
    f1_score = 2 * recall * precision / (precision + recall + 1.0e-16)
    best_idx = np.argmax(f1_score)
    f1_score_message = (
        f"Best f1: {f1_score[best_idx]:.5f},\n"
        f"recall: {recall[best_idx]:.5f},\n"
        f"precision: {precision[best_idx]:.5f},\n"
        f"threshold: {thresholds[best_idx]:.5f}"
    )
    if verbose:
        print(f1_score_message)

    plt.plot(recall, precision, marker=".", label=f"AUPRC={pr_auc:.5f}")
    plt.scatter(
        recall[best_idx],
        precision[best_idx],
        marker="o",
        color="black",
        label=f1_score_message,
        linewidths=5,
    )
    plt.xlabel("Recall")
    plt.xlim(0, 1)
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend()

    if path:
        plt.savefig(path)

    if plot:
        plt.show()
    plt.close()

    return pr_auc
