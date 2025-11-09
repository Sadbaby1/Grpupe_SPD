#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 7 — Hyperparameter Tuning of a Deep MLP on the MNIST Dataset Using Keras Tuner

What this script does
---------------------
1) Loads MNIST, normalizes to [0,1], flattens to 784 features.
2) Builds a tunable MLP with Keras Tuner (RandomSearch).
   Tuned hyperparameters:
     - num_layers     (2..5)
     - units per layer (64..512, step 64)
     - dropout        (0.0..0.5, step 0.1)
     - learning rate  ({1e-2, 5e-3, 1e-3, 5e-4, 1e-4})
     - batch size     ({32, 64, 128})
3) Runs tuner.search with EarlyStopping.
4) Rebuilds best model, retrains on train+val, evaluates on test.
5) Saves:
     - best learning curves (PNG)
     - tuner results summary (CSV)
     - best hyperparameters (JSON/TXT)
     - model training log (CSV)
     - trained model (Keras .keras file)
     - a short README-like text summary

Outputs are written to: ./assignment7_outputs
"""

import os, json, time, csv, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- Try to import keras-tuner ---
try:
    import keras_tuner as kt
except Exception:
    print("\n[INFO] Installing keras-tuner ...")
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", "keras-tuner"])
    import keras_tuner as kt

# ------------------ Config ------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTDIR = "assignment7_outputs"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

MAX_TRIALS = 15          # Increase to 30–50 for a deeper search if you have time
EPOCHS_TUNE = 20         # Initial tuning epochs
EPOCHS_FINAL = 30        # Retrain best model
PATIENCE = 5             # Early stopping patience

# ------------------ Data ------------------
print("[1/6] Loading MNIST ...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# normalize & flatten
x_train = (x_train.astype("float32") / 255.0).reshape(len(x_train), -1)
x_test  = (x_test.astype("float32")  / 255.0).reshape(len(x_test), -1)

# Hold out a validation split from training for tuning
VAL_SIZE = 10000
x_val   = x_train[-VAL_SIZE:]
y_val   = y_train[-VAL_SIZE:]
x_train = x_train[:-VAL_SIZE]
y_train = y_train[:-VAL_SIZE]

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ------------------ HyperModel ------------------
class MNISTHyperModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        inputs = keras.Input(shape=(784,), name="pixel_input")
        x = inputs

        num_layers = hp.Int("num_layers", min_value=2, max_value=5, step=1)
        dropout_rate = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)

        for i in range(num_layers):
            units = hp.Int(f"units_{i}", min_value=64, max_value=512, step=64)
            x = layers.Dense(units, activation="relu")(x)
            if dropout_rate > 0.0:
                x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Dense(10, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="mlp_mnist_tunable")

        lr = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
        opt = keras.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=opt,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    # Tune batch_size by overriding fit()
    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", values=[32, 64, 128])
        return model.fit(*args, batch_size=batch_size, **kwargs)

hypermodel = MNISTHyperModel()

# ------------------ Tuner ------------------
print("[2/6] Setting up Keras Tuner (RandomSearch) ...")
tuner = kt.RandomSearch(
    hypermodel=hypermodel,
    objective="val_accuracy",
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory=OUTDIR,
    project_name="kt_mnist_mlp",
    overwrite=True
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max",
    patience=PATIENCE, restore_best_weights=True
)

print("[3/6] Starting hyperparameter search ...")
t0 = time.time()
tuner.search(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS_TUNE,
    callbacks=[early_stop],
    verbose=1
)
search_time_s = time.time() - t0
print(f"[TUNER] Finished in {search_time_s/60:.1f} minutes.")

# Save tuner results summary CSV (trial -> val_accuracy)
results_path = os.path.join(OUTDIR, "tuner_trial_results.csv")
with open(results_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["trial_id", "score", "hyperparameters"])
    for t in tuner.oracle.get_best_trials(num_trials=MAX_TRIALS):
        writer.writerow([t.trial_id, t.score, json.dumps(t.hyperparameters.values)])

# ------------------ Best HPs & Re-train ------------------
print("[4/6] Extracting best hyperparameters ...")
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hp_path = os.path.join(OUTDIR, "best_hyperparameters.json")
with open(best_hp_path, "w", encoding="utf-8") as f:
    json.dump(best_hp.values, f, indent=2)
print("Best HP:", best_hp.values)

# Rebuild with best HP and train on Train+Val
print("[5/6] Retraining best model on train+val ...")
x_train_full = np.concatenate([x_train, x_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

best_model = hypermodel.build(best_hp)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=PATIENCE, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", mode="max",
        factor=0.5, patience=3, min_lr=1e-5, verbose=1
    ),
    keras.callbacks.CSVLogger(os.path.join(OUTDIR, "final_training_log.csv"))
]

t1 = time.time()
history = best_model.fit(
    x_train_full, y_train_full,
    validation_split=0.1,       # small internal validation for callbacks
    epochs=EPOCHS_FINAL,
    batch_size=best_hp.get("batch_size"),
    callbacks=callbacks,
    verbose=1
)
final_train_time_s = time.time() - t1

# ------------------ Evaluate & Save ------------------
print("[6/6] Evaluating on test set ...")
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy = {test_acc:.4f} | Test loss = {test_loss:.4f}")

# Save model
model_path = os.path.join(OUTDIR, "best_mlp_mnist.keras")
best_model.save(model_path)

# Plot learning curves
import matplotlib.pyplot as plt

def plot_learning_curves(hist, out_png):
    epochs = range(1, len(hist["accuracy"])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, hist["accuracy"], label="Train acc")
    plt.plot(epochs, hist["val_accuracy"], label="Val acc")
    plt.plot(epochs, hist["loss"], label="Train loss")
    plt.plot(epochs, hist["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.title("Learning Curves — Best MLP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

plot_learning_curves(history.history, os.path.join(OUTDIR, "learning_curves_best_mlp.png"))

# Write a quick summary file (handy for the report)
summary_txt = os.path.join(OUTDIR, "summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Assignment 7 — MNIST MLP with Keras Tuner\n")
    f.write("="*55 + "\n\n")
    f.write(f"Search time (tuner): {search_time_s:.1f}s\n")
    f.write(f"Final train time:     {final_train_time_s:.1f}s\n")
    f.write(f"Test accuracy:        {test_acc:.4f}\n")
    f.write("\nBest Hyperparameters:\n")
    for k, v in best_hp.values.items():
        f.write(f" - {k}: {v}\n")

print("\nArtifacts saved to:", OUTDIR)
print("Files created:")
for name in [
    "tuner_trial_results.csv",
    "best_hyperparameters.json",
    "final_training_log.csv",
    "learning_curves_best_mlp.png",
    "best_mlp_mnist.keras",
    "summary.txt"
]:
    print(" -", name)

