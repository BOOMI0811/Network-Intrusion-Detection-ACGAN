"""End-to-end pipeline:  baseline  ->  train ACGAN  ->  augment  ->  retrain  ->  compare.

Usage (from repo root):
    python -m src.train
    python -m src.train --gan-epochs 200 --target-ratio 0.5
"""
import argparse
import json

import numpy as np
import tensorflow as tf

from . import config as C
from .acgan import ACGAN
from .classifier import build_classifier
from .data_preprocessing import load_nsl_kdd
from .evaluate import (evaluate, plot_class_distribution, plot_comparison,
                       plot_confusion_matrix, plot_gan_loss)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gan-epochs", type=int, default=C.GAN_EPOCHS)
    p.add_argument("--clf-epochs", type=int, default=C.CLF_EPOCHS)
    p.add_argument("--target-ratio", type=float, default=C.TARGET_RATIO,
                   help="oversample each class up to ratio * (largest class size)")
    return p.parse_args()


def train_classifier(X, y, n_classes, epochs):
    idx = np.random.permutation(len(X))  # shuffle so validation_split isn't just the tail
    X, y = X[idx], y[idx]
    model = build_classifier(X.shape[1], n_classes)
    model.fit(X, y, epochs=epochs, batch_size=C.CLF_BATCH_SIZE,
              validation_split=0.1, verbose=2)
    return model


def main():
    args = parse_args()
    np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    for d in (C.RESULTS_DIR, C.MODELS_DIR):
        d.mkdir(exist_ok=True)

    print("Loading NSL-KDD ...")
    X_train, y_train, X_test, y_test = load_nsl_kdd()
    n_features, n_classes = X_train.shape[1], len(C.CLASS_NAMES)
    counts = np.bincount(y_train, minlength=n_classes)
    print("Train class counts:", dict(zip(C.CLASS_NAMES, counts.tolist())))

    # 1) Baseline on imbalanced data
    print("\n=== Baseline classifier (no augmentation) ===")
    baseline_model = train_classifier(X_train, y_train, n_classes, args.clf_epochs)
    baseline = evaluate(baseline_model, X_test, y_test)
    print(f"Baseline accuracy: {baseline['accuracy']:.4f} | "
          f"normal/attack: {baseline['binary_accuracy']:.4f}")

    # 2) Train ACGAN
    print("\n=== Training ACGAN ===")
    gan = ACGAN(n_features, n_classes, C.LATENT_DIM)
    history = gan.fit(X_train, y_train, epochs=args.gan_epochs, batch_size=C.GAN_BATCH_SIZE)
    gan.generator.save(C.MODELS_DIR / "acgan_generator.keras")
    gan.discriminator.save(C.MODELS_DIR / "acgan_discriminator.keras")

    # 3) Generate synthetic samples for under-represented classes
    target = int(counts.max() * args.target_ratio)
    X_syn, y_syn = [], []
    for cid in range(n_classes):
        need = target - counts[cid]
        if need > 0:
            print(f"Generating {need} synthetic '{C.CLASS_NAMES[cid]}' samples")
            X_syn.append(gan.generate(cid, need))
            y_syn.append(np.full(need, cid, dtype="int32"))
    if X_syn:
        X_aug = np.vstack([X_train] + X_syn)
        y_aug = np.concatenate([y_train] + y_syn)
    else:
        X_aug, y_aug = X_train, y_train
    new_counts = np.bincount(y_aug, minlength=n_classes)

    # 4) Retrain the same classifier on augmented data
    print("\n=== Classifier with ACGAN-augmented data ===")
    aug_model = train_classifier(X_aug, y_aug, n_classes, args.clf_epochs)
    augmented = evaluate(aug_model, X_test, y_test)
    print(f"Augmented accuracy: {augmented['accuracy']:.4f} | "
          f"normal/attack: {augmented['binary_accuracy']:.4f}")

    # 5) Save results
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({"baseline": baseline, "augmented": augmented,
                   "class_counts_before": counts.tolist(),
                   "class_counts_after": new_counts.tolist()}, f, indent=2)
    plot_confusion_matrix(baseline["confusion_matrix"], "Baseline",
                          C.RESULTS_DIR / "confusion_matrix_baseline.png")
    plot_confusion_matrix(augmented["confusion_matrix"], "With ACGAN data",
                          C.RESULTS_DIR / "confusion_matrix_acgan.png")
    plot_comparison(baseline, augmented, C.RESULTS_DIR / "accuracy_comparison.png")
    plot_class_distribution(counts, new_counts, C.RESULTS_DIR / "class_distribution.png")
    plot_gan_loss(history, C.RESULTS_DIR / "gan_loss.png")
    print("\nDone. Results saved in results/")


if __name__ == "__main__":
    main()
