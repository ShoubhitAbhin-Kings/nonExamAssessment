import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# === PARAMETERS TO EDIT ===
model_path = 'CNNModels/model6.keras'        # Path to your .keras model
val_data_dir = 'savedDataForAllModels/dataForsign_language_model/trainOnThese/temporaryForConfusionMatrixA'  # Your validation directory
target_class = 'A'                     # The class you want to compute TP, FP, FN, TN for
image_size = (300, 300)                # Resize if needed
batch_size = 32

# === LOAD MODEL ===
model = tf.keras.models.load_model(model_path)

# === LOAD VALIDATION DATA ===
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',  # to match softmax output
    shuffle=False
)

class_names = val_ds.class_names
target_index = class_names.index(target_class)

# === MAKE PREDICTIONS ===
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels.numpy(), axis=1)
    y_true.extend(true_labels)
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === BINARY CONFUSION MATRIX ===
# 1 = target class, 0 = any other
binary_true = (y_true == target_index).astype(int)
binary_pred = (y_pred == target_index).astype(int)

tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()

print(f"Class: {target_class}")
print(f"True Positive (TP): {tp}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Negative (TN): {tn}")