"""
train.py  –  SignBridge ISL CNN Training
Uses Keras ImageDataGenerator with validation_split directly on the
original dataset folder (no file copying required).

Dataset layout expected:
    dataset/
        a/   (1200 images of sign A)
        b/   (1200 images of sign B)
        ...
        z/   (1200 images of sign Z)
        {/   (1200 images of SPACE sign)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all systems)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
MODELS_DIR  = "models"
DOCS_DIR    = "docs"
for d in [MODELS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
IMG_SIZE      = 64
BATCH_SIZE    = 64
EPOCHS        = 30
VAL_SPLIT     = 0.2
SEED          = 42

# ── Dataset check ──────────────────────────────────────────────────────────────
def check_dataset():
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] '{DATASET_DIR}' directory not found.")
        return False
    subdirs = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not subdirs:
        print(f"[ERROR] '{DATASET_DIR}' has no class subdirectories.")
        return False

    total = 0
    print("\n[+] Dataset class summary:")
    for cls in sorted(subdirs):
        cls_path = os.path.join(DATASET_DIR, cls)
        n = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        label = cls.upper() if cls != '{' else 'SPACE'
        print(f"    {label:6s} : {n} images")
        total += n
    print(f"\n    Total images : {total}\n")
    return total > 0

# ── Model architecture ─────────────────────────────────────────────────────────
def build_model(num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classifier head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    gpus = tf.config.list_physical_devices('GPU')
    print("=" * 55)
    print(f"  SignBridge ISL Training   TF {tf.__version__}")
    print("=" * 55)
    print(f"  GPUs available : {len(gpus)}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  Using GPU      : {gpus[0].name}")
    else:
        print("  Running on CPU.")

    if not check_dataset():
        return

    # ── Data generators (NO file copying) ────────────────────────────────────
    # Both generators share the same rescale; augmentation only on train subset.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,   # ISL signs are directional
        fill_mode='nearest',
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
    )

    print("[>>] Creating data generators (no file copying)...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=SEED,
    )

    # ── Save class-label mapping ──────────────────────────────────────────────
    # idx_to_label: {0: 'A', 1: 'B', ..., 26: 'SPACE'}
    idx_to_label = {}
    for label, idx in train_gen.class_indices.items():
        display = 'SPACE' if label == '{' else label.upper()
        idx_to_label[str(idx)] = display

    labels_path = os.path.join(MODELS_DIR, "class_labels.json")
    with open(labels_path, "w") as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"[+] Class labels saved  : '{labels_path}'")
    print(f"[+] Classes detected    : {sorted(train_gen.class_indices.keys())}")
    print(f"[+] Num classes         : {len(train_gen.class_indices)}")
    print(f"[+] Train batches       : {len(train_gen)}")
    print(f"[+] Val   batches       : {len(val_gen)}")

    num_classes = len(train_gen.class_indices)

    # ── Build & summarise ─────────────────────────────────────────────────────
    model = build_model(num_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_path = os.path.join(MODELS_DIR, "signbridge_best.h5")
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            best_path, monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[>>] Starting training for up to {EPOCHS} epochs ...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n[OK] Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"     Validation Loss     : {val_loss:.4f}")

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "signbridge_cnn.h5")
    model.save(final_path)
    print(f"\n[SAVED] Final model : '{final_path}'")
    print(f"        Best model  : '{best_path}'")

    # ── Training plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Acc',  linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc',    linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plot_path = os.path.join(DOCS_DIR, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training plot saved : '{plot_path}'")


if __name__ == "__main__":
    main()
