"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import CLASS_NAMES


def evaluate(model, X, y):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        # normal (0) vs attack (anything else)
        "binary_accuracy": float(accuracy_score(y != 0, pred != 0)),
        "report": classification_report(y, pred, labels=labels, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_comparison(baseline, augmented, path):
    names = ["Multi-class acc.", "Normal/Attack acc."]
    b = [baseline["accuracy"] * 100, baseline["binary_accuracy"] * 100]
    a = [augmented["accuracy"] * 100, augmented["binary_accuracy"] * 100]
    x = np.arange(len(names))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, b, 0.4, label="Baseline")
    plt.bar(x + 0.2, a, 0.4, label="With ACGAN data")
    plt.xticks(x, names); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
    plt.title("Baseline vs ACGAN-augmented"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_class_distribution(before, after, path):
    x = np.arange(len(CLASS_NAMES))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, before, 0.4, label="Original")
    plt.bar(x + 0.2, after, 0.4, label="After ACGAN")
    plt.xticks(x, CLASS_NAMES); plt.yscale("log"); plt.ylabel("Samples (log scale)")
    plt.title("Class distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_gan_loss(history, path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["d_loss"], label="Discriminator")
    plt.plot(history["g_loss"], label="Generator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ACGAN training loss"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
