# =========================================================
# ECG ARRHYTHMIA MULTICLASS CLASSIFICATION
# FINAL STABLE VERSION
#
# Classes:
# N = Normal
# S = Supraventricular
# V = Ventricular
#
# Features:
# - Inter-patient evaluation
# - CNN + BiLSTM
# - RR interval features
# - Targeted augmentation
# - Multiclass focal loss
# - GPU training enabled
# - Full visualization suite
#
# =========================================================

import os
import json
import random
import datetime
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Dense,
    Bidirectional,
    LSTM,
    Concatenate
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize


# =========================================================
# 1. REPRODUCIBILITY
# =========================================================

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

# GPU deterministic operations
tf.config.experimental.enable_op_determinism()


# =========================================================
# 2. GPU CHECK
# =========================================================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\nGPU detected and enabled.")
    print(gpus)

else:
    print("\nNo GPU detected. Running on CPU.")


# =========================================================
# 3. SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "ECG_DATA_DIR",
    os.path.join(
        BASE_DIR,
        "data",
        "mit-bih-arrhythmia-database-1.0.0"
    )
)

if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}"
    )

RUN_NAME = os.environ.get(
    "ECG_RUN_NAME",
    "baseline"
)

OUT_ROOT = os.environ.get(
    "ECG_OUT_DIR",
    os.path.join(BASE_DIR, "runs")
)

RUN_DIR = os.path.join(
    OUT_ROOT,
    RUN_NAME
)

os.makedirs(
    RUN_DIR,
    exist_ok=True
)

print(f"\nDATA_DIR: {DATA_DIR}")
print(f"RUN_DIR : {RUN_DIR}")

# Inter-patient split
DS1 = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]

DS2 = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

PRE_SAMPLES = 90
POST_SAMPLES = 144

SEGMENT_LENGTH = PRE_SAMPLES + POST_SAMPLES

LEAD_INDEX = 0

BATCH_SIZE = 128
EPOCHS = 40


# =========================================================
# 4. AAMI MAPPING
# =========================================================

AAMI_MAP = {

    'N': 'N',
    'L': 'N',
    'R': 'N',
    'e': 'N',
    'j': 'N',

    'A': 'S',
    'a': 'S',
    'J': 'S',
    'S': 'S',

    'V': 'V',
    'E': 'V'
}


# =========================================================
# 5. LABEL ENCODING
# =========================================================

LABEL_TO_INT = {
    'N': 0,
    'S': 1,
    'V': 2
}

INT_TO_LABEL = {
    0: 'N',
    1: 'S',
    2: 'V'
}

NUM_CLASSES = 3


# =========================================================
# 6. NORMALIZATION
# =========================================================

def normalize_segment(segment):

    mean = np.mean(segment)
    std = np.std(segment)

    if std == 0:
        return segment - mean

    return (segment - mean) / std


def normalize_rr(rr_array):

    rr_array = np.array(rr_array, dtype=np.float32)

    mean = np.mean(rr_array)
    std = np.std(rr_array)

    if std == 0:
        return rr_array - mean

    return (rr_array - mean) / std


# =========================================================
# 7. EXTRACT ECG + RR FEATURES
# =========================================================

def extract_beats_from_record(
    record_name,
    data_dir,
    pre_samples,
    post_samples,
    lead_index=0
):

    record_path = os.path.join(
        data_dir,
        record_name
    )

    try:

        signal_record = wfdb.rdrecord(record_path)

        annotation = wfdb.rdann(
            record_path,
            'atr'
        )

    except Exception as e:

        print(f"Error reading {record_name}: {e}")

        return [], [], []

    signal = signal_record.p_signal[:, lead_index]

    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    beats = []
    labels = []
    rr_features = []

    for i in range(1, len(ann_samples) - 1):

        r_peak = ann_samples[i]

        symbol = ann_symbols[i]

        if symbol not in AAMI_MAP:
            continue

        start = r_peak - pre_samples
        end = r_peak + post_samples

        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]

        if len(segment) != SEGMENT_LENGTH:
            continue

        prev_rr = ann_samples[i] - ann_samples[i - 1]
        next_rr = ann_samples[i + 1] - ann_samples[i]

        rr = [prev_rr, next_rr]

        segment = normalize_segment(segment)

        beats.append(segment)

        labels.append(
            AAMI_MAP[symbol]
        )

        rr_features.append(rr)

    return beats, labels, rr_features


# =========================================================
# 8. LOAD DATASET
# =========================================================

def load_dataset(
    record_list,
    data_dir,
    pre_samples,
    post_samples,
    lead_index=0
):

    all_beats = []
    all_labels = []
    all_rr = []

    for rec in record_list:

        print(f"Loading record {rec} ...")

        beats, labels, rr = extract_beats_from_record(
            rec,
            data_dir,
            pre_samples,
            post_samples,
            lead_index
        )

        all_beats.extend(beats)

        all_labels.extend(labels)

        all_rr.extend(rr)

    X = np.array(
        all_beats,
        dtype=np.float32
    )

    y = np.array(all_labels)

    RR = np.array(
        all_rr,
        dtype=np.float32
    )

    return X, y, RR


# =========================================================
# 9. ECG AUGMENTATION
# =========================================================

def augment_segment(segment):

    x = segment.copy()

    # Amplitude scaling
    if np.random.rand() < 0.8:

        scale = np.random.uniform(
            0.90,
            1.10
        )

        x = x * scale

    # Time shift
    if np.random.rand() < 0.8:

        shift = np.random.randint(-5, 6)

        if shift != 0:

            x = np.roll(x, shift)

            if shift > 0:
                x[:shift] = x[shift]

            else:
                x[shift:] = x[shift - 1]

    # Gaussian noise
    if np.random.rand() < 0.9:

        noise = np.random.normal(
            0,
            0.01,
            size=x.shape
        )

        x = x + noise

    x = normalize_segment(x)

    return x.astype(np.float32)


# =========================================================
# 10. TARGETED AUGMENTATION
# =========================================================

def augment_training_data(
    X,
    RR,
    y
):

    X_list = [X]
    RR_list = [RR]
    y_list = [y]

    for sample, rr, label in zip(X, RR, y):

        segment = sample.squeeze(-1)

        # N
        if label == 0:
            multiplier = 0

        # S
        elif label == 1:
            multiplier = 6

        # V
        elif label == 2:
            multiplier = 2

        else:
            multiplier = 0

        for _ in range(multiplier):

            aug = augment_segment(segment)

            aug = np.expand_dims(
                aug,
                axis=-1
            )

            X_list.append(
                aug[np.newaxis, ...]
            )

            RR_list.append(
                rr[np.newaxis, ...]
            )

            y_list.append(
                np.array(
                    [label],
                    dtype=np.int32
                )
            )

    X_new = np.concatenate(
        X_list,
        axis=0
    )

    RR_new = np.concatenate(
        RR_list,
        axis=0
    )

    y_new = np.concatenate(
        y_list,
        axis=0
    )

    idx = np.random.permutation(
        len(y_new)
    )

    return (
        X_new[idx],
        RR_new[idx],
        y_new[idx]
    )


# =========================================================
# 11. MULTICLASS FOCAL LOSS
# =========================================================

def categorical_focal_loss(
    alpha=0.50,
    gamma=2.0
):

    def loss(y_true, y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow(
            1 - y_pred,
            gamma
        )

        focal_loss = weight * cross_entropy

        return tf.reduce_mean(
            tf.reduce_sum(
                focal_loss,
                axis=1
            )
        )

    return loss


# =========================================================
# 12. LOAD TRAIN / TEST
# =========================================================

print("Loading DS1 (train) ...")

X_train, y_train, RR_train = load_dataset(
    DS1,
    DATA_DIR,
    PRE_SAMPLES,
    POST_SAMPLES,
    LEAD_INDEX
)

print("\nLoading DS2 (test) ...")

X_test, y_test, RR_test = load_dataset(
    DS2,
    DATA_DIR,
    PRE_SAMPLES,
    POST_SAMPLES,
    LEAD_INDEX
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("\nOriginal Train Distribution:")
print(Counter(y_train))

print("\nOriginal Test Distribution:")
print(Counter(y_test))


# =========================================================
# 13. CLASS DISTRIBUTION PLOT
# =========================================================

train_counter = Counter(y_train)

plt.figure(figsize=(6, 5))

plt.bar(
    train_counter.keys(),
    train_counter.values()
)

plt.title('Training Class Distribution')

plt.xlabel('Class')

plt.ylabel('Count')

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'class_distribution.png'),
    dpi=300
)

plt.show()


# =========================================================
# 14. ENCODE LABELS
# =========================================================

y_train_encoded = np.array([
    LABEL_TO_INT[label]
    for label in y_train
], dtype=np.int32)

y_test_encoded = np.array([
    LABEL_TO_INT[label]
    for label in y_test
], dtype=np.int32)


# =========================================================
# 15. NORMALIZE RR
# =========================================================

RR_train = normalize_rr(RR_train)
RR_test = normalize_rr(RR_test)


# =========================================================
# 16. PREPARE CNN INPUT
# =========================================================

X_train = np.expand_dims(
    X_train,
    axis=-1
)

X_test = np.expand_dims(
    X_test,
    axis=-1
)


# =========================================================
# 17. TRAIN / VALIDATION SPLIT
# =========================================================

(
    X_tr,
    X_val,
    RR_tr,
    RR_val,
    y_tr,
    y_val

) = train_test_split(

    X_train,
    RR_train,
    y_train_encoded,

    test_size=0.1,

    random_state=SEED,

    stratify=y_train_encoded
)


# =========================================================
# 18. AUGMENT TRAINING DATA
# =========================================================

X_tr_aug, RR_tr_aug, y_tr_aug = augment_training_data(
    X_tr,
    RR_tr,
    y_tr
)


# =========================================================
# 19. ONE HOT ENCODING
# =========================================================

y_tr_aug_cat = tf.keras.utils.to_categorical(
    y_tr_aug,
    num_classes=NUM_CLASSES
)

y_val_cat = tf.keras.utils.to_categorical(
    y_val,
    num_classes=NUM_CLASSES
)

y_test_cat = tf.keras.utils.to_categorical(
    y_test_encoded,
    num_classes=NUM_CLASSES
)


# =========================================================
# 20. BUILD MODEL
# =========================================================

def build_model(ecg_shape, rr_shape):

    # ECG INPUT
    ecg_input = Input(
        shape=ecg_shape,
        name='ecg_input'
    )

    # CNN BLOCK 1
    x = Conv1D(
        32,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(ecg_input)

    x = BatchNormalization()(x)

    x = Conv1D(
        32,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(x)

    # CNN BLOCK 2
    x = Conv1D(
        64,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = Conv1D(
        64,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.25)(x)

    # CNN BLOCK 3
    x = Conv1D(
        128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = Conv1D(
        128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.3)(x)

    # BiLSTM
    x = Bidirectional(
        LSTM(
            64,
            return_sequences=False
        )
    )(x)

    x = Dropout(0.4)(x)

    # RR INPUT
    rr_input = Input(
        shape=rr_shape,
        name='rr_input'
    )

    rr_branch = Dense(
        16,
        activation='relu'
    )(rr_input)

    rr_branch = Dropout(0.2)(rr_branch)

    # CONCATENATE
    combined = Concatenate()([
        x,
        rr_branch
    ])

    combined = Dense(
        128,
        activation='relu'
    )(combined)

    combined = Dropout(0.5)(combined)

    combined = Dense(
        64,
        activation='relu'
    )(combined)

    combined = Dropout(0.4)(combined)

    # OUTPUT
    output = Dense(
        NUM_CLASSES,
        activation='softmax'
    )(combined)

    model = Model(
        inputs=[ecg_input, rr_input],
        outputs=output
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss=categorical_focal_loss(
            alpha=0.50,
            gamma=2.0
        ),

        metrics=['accuracy']
    )

    return model


model = build_model(
    ecg_shape=X_tr_aug.shape[1:],
    rr_shape=(2,)
)

model.summary()


# =========================================================
# 21. CALLBACKS
# =========================================================

callbacks = [

    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]


# =========================================================
# 22. TRAIN MODEL
# =========================================================

history = model.fit(

    [X_tr_aug, RR_tr_aug],
    y_tr_aug_cat,

    validation_data=(
        [X_val, RR_val],
        y_val_cat
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# =========================================================
# 23. TRAINING CURVES
# =========================================================

plt.figure(figsize=(12, 5))

# LOSS
plt.subplot(1, 2, 1)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

# ACCURACY
plt.subplot(1, 2, 2)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Training and Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'training_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 24. TEST EVALUATION
# =========================================================

y_pred_prob = model.predict(
    [X_test, RR_test],
    batch_size=BATCH_SIZE
)

y_pred_enc = np.argmax(
    y_pred_prob,
    axis=1
)

acc = accuracy_score(
    y_test_encoded,
    y_pred_enc
)

print(f"\nTest Accuracy: {acc:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test_encoded,
    y_pred_enc,
    target_names=[
        'N',
        'S',
        'V'
    ],
    digits=4
))

cm = confusion_matrix(
    y_test_encoded,
    y_pred_enc
)

print("\nConfusion Matrix:\n")
print(cm)


# =========================================================
# 25. CONFUSION MATRIX VISUALIZATION
# =========================================================

classes = ['N', 'S', 'V']

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'confusion_matrix.png'),
    dpi=300
)

plt.show()


# =========================================================
# 26. ROC CURVES
# =========================================================

y_test_bin = label_binarize(
    y_test_encoded,
    classes=[0, 1, 2]
)

plt.figure(figsize=(8, 6))

per_class_roc_auc = {}

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_pred_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    per_class_roc_auc[INT_TO_LABEL[i]] = float(roc_auc)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {i} AUC = {roc_auc:.3f}'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curves')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'roc_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 27. PRECISION-RECALL CURVES
# =========================================================

plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):

    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_pred_prob[:, i]
    )

    plt.plot(
        recall,
        precision,
        label=f'Class {i}'
    )

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curves')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'precision_recall_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 28. SAMPLE ECG VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 6))

for i in range(3):

    plt.subplot(3, 1, i + 1)

    plt.plot(
        X_test[i].squeeze()
    )

    plt.title(
        f'True Label: {INT_TO_LABEL[y_test_encoded[i]]}'
    )

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'sample_ecg_signals.png'),
    dpi=300
)

plt.show()


# =========================================================
# 29. SAVE MODEL
# =========================================================

MODEL_PATH = os.path.join(
    RUN_DIR,
    "best_ecg_multiclass_model.keras"
)

model.save(MODEL_PATH)

print(
    f"\nModel saved as {MODEL_PATH}"
)


# =========================================================
# 30. SAVE METRICS AND HISTORY
# =========================================================

metrics = {

    "run_name": RUN_NAME,

    "timestamp": datetime.datetime.now().isoformat(),

    "tensorflow_version": tf.__version__,

    "gpus": [str(gpu) for gpu in gpus],

    "config": {

        "SEED": SEED,

        "EPOCHS": EPOCHS,

        "BATCH_SIZE": BATCH_SIZE,

        "PRE_SAMPLES": PRE_SAMPLES,

        "POST_SAMPLES": POST_SAMPLES,

        "SEGMENT_LENGTH": SEGMENT_LENGTH,

        "LEAD_INDEX": LEAD_INDEX,

        "focal_loss_alpha": 0.50,

        "focal_loss_gamma": 2.0,

        "adam_learning_rate": 1e-4,

        "DS1": DS1,

        "DS2": DS2
    },

    "train_distribution": {
        str(label): int(count)
        for label, count in Counter(y_train).items()
    },

    "test_distribution": {
        str(label): int(count)
        for label, count in Counter(y_test).items()
    },

    "test_accuracy": float(acc),

    "classification_report": classification_report(
        y_test_encoded,
        y_pred_enc,
        target_names=[
            'N',
            'S',
            'V'
        ],
        digits=4,
        output_dict=True
    ),

    "confusion_matrix": cm.tolist(),

    "per_class_roc_auc": per_class_roc_auc
}

METRICS_PATH = os.path.join(
    RUN_DIR,
    "metrics.json"
)

with open(METRICS_PATH, "w") as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

print(f"Metrics saved as {METRICS_PATH}")

HISTORY_PATH = os.path.join(
    RUN_DIR,
    "history.json"
)

history_serializable = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w") as f:

    json.dump(
        history_serializable,
        f,
        indent=2
    )

print(f"History saved as {HISTORY_PATH}")