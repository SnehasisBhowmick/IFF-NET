import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, callbacks, optimizers #type: ignore
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import csv
import keras

def load_image(path, target_size=(128, 128)):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0

def load_mask(path, target_size=(128, 128)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = mask / 255.0
    return np.expand_dims(mask, axis=-1)

image_dir = "E:/Pyhton for wound segmentation/wound-segmentation-master/wound-segmentation-master/data/wound_dataset/azh_wound_care_center_dataset_patches/train/images_train"
mask_dir = "E:/Pyhton for wound segmentation/wound-segmentation-master/wound-segmentation-master/data/wound_dataset/azh_wound_care_center_dataset_patches/train/labels_train"

image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

X, Y = [], []
for img_name in tqdm(image_list, desc="Loading data"):
    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)
    if os.path.exists(img_path) and os.path.exists(mask_path):
        img = load_image(img_path)
        mask = load_mask(mask_path)
        if img is not None and mask is not None:
            X.append(img)
            Y.append(mask)
    else:
        print(f"Skipping {img_name}, mask or image missing.")

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
print("Loaded images shape:", X.shape)
print("Loaded masks shape:", Y.shape)

# --- Dice loss and metric ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

from tensorflow.keras.layers import Dropout, LeakyReLU #type: ignore
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dropout, LeakyReLU #type: ignore
import tensorflow_addons as tfa
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, concatenate, LeakyReLU

input_shape = (128, 128, 3)
input = Input(input_shape)

# Layer 1
c1_concat1 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(input)
c1_concat1 = tfa.layers.GroupNormalization(groups=4)(c1_concat1)
c1_concat1 = LeakyReLU(alpha=0.1)(c1_concat1)
c1_concat1 = Dropout(0.2)(c1_concat1)

c1_concat2 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(c1_concat1)
c1_concat2 = tfa.layers.GroupNormalization(groups=4)(c1_concat2)
c1_concat2 = LeakyReLU(alpha=0.1)(c1_concat2)
c1_concat2 = Dropout(0.2)(c1_concat2)

c1_out = concatenate([c1_concat1, c1_concat2])
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1_out)

# Layer 2
c2_concat1 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(p1)
c2_concat1 = tfa.layers.GroupNormalization(groups=4)(c2_concat1)
c2_concat1 = LeakyReLU(alpha=0.1)(c2_concat1)
c2_concat1 = Dropout(0.2)(c2_concat1)

c2_concat2 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(c2_concat1)
c2_concat2 = tfa.layers.GroupNormalization(groups=4)(c2_concat2)
c2_concat2 = LeakyReLU(alpha=0.1)(c2_concat2)
c2_concat2 = Dropout(0.2)(c2_concat2)

c2_out = concatenate([c2_concat1, c2_concat2])
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2_out)

# Layer 3
c3_concat1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(p2)
c3_concat1 = tfa.layers.GroupNormalization(groups=4)(c3_concat1)
c3_concat1 = LeakyReLU(alpha=0.1)(c3_concat1)
c3_concat1 = Dropout(0.2)(c3_concat1)

c3_concat2 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(c3_concat1)
c3_concat2 = tfa.layers.GroupNormalization(groups=4)(c3_concat2)
c3_concat2 = LeakyReLU(alpha=0.1)(c3_concat2)
c3_concat2 = Dropout(0.2)(c3_concat2)

c3_out = concatenate([c3_concat1, c3_concat2])
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3_out)

# Layer 4
c4_concat1 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(p3)
c4_concat1 = tfa.layers.GroupNormalization(groups=4)(c4_concat1)
c4_concat1 = LeakyReLU(alpha=0.1)(c4_concat1)
c4_concat1 = Dropout(0.2)(c4_concat1)

c4_concat2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(c4_concat1)
c4_concat2 = tfa.layers.GroupNormalization(groups=4)(c4_concat2)
c4_concat2 = LeakyReLU(alpha=0.1)(c4_concat2)
c4_concat2 = Dropout(0.2)(c4_concat2)

c4_out = concatenate([c4_concat1, c4_concat2])
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4_out)

# Bottleneck
c5 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(p4)
c5 = tfa.layers.GroupNormalization(groups=4)(c5)
c5 = LeakyReLU(alpha=0.1)(c5)

c5 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(c5)
c5 = tfa.layers.GroupNormalization(groups=4)(c5)
c5 = LeakyReLU(alpha=0.1)(c5)

# Decoder Layer 4
u1 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u1 = concatenate([u1, c4_concat2])
u1 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(u1)
u1 = tfa.layers.GroupNormalization(groups=4)(u1)
u1 = LeakyReLU(alpha=0.1)(u1)
u1 = Dropout(0.2)(u1)

# Decoder Layer 3
u2 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(u1)
u2 = concatenate([u2, c3_concat2])
u2 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(u2)
u2 = tfa.layers.GroupNormalization(groups=4)(u2)
u2 = LeakyReLU(alpha=0.1)(u2)
u2 = Dropout(0.2)(u2)

# Decoder Layer 2
u3 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(u2)
u3 = concatenate([u3, c2_concat2])
u3 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(u3)
u3 = tfa.layers.GroupNormalization(groups=4)(u3)
u3 = LeakyReLU(alpha=0.1)(u3)
u3 = Dropout(0.1)(u3)

# Decoder Layer 1
u4 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(u3)
u4 = concatenate([u4, c1_concat2])
u4 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(u4)
u4 = tfa.layers.GroupNormalization(groups=4)(u4)
u4 = LeakyReLU(alpha=0.1)(u4)

output = Conv2D(1, (1,1), activation='sigmoid')(u4)

model = keras.Model(inputs=[input], outputs=[output])

model.summary()

optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss=dice_loss,
    metrics=['accuracy', dice_coefficient],
    run_eagerly=False
)

import numpy as np

def cosine_annealing(epoch, lr):
    max_epochs = 100
    min_lr = 1e-7
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    lr = (1e-3 - min_lr) * cosine_decay + min_lr
    return lr

lr_scheduler = callbacks.LearningRateScheduler(cosine_annealing, verbose=1)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=False, verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1
)

cb = [early_stopping, reduce_lr, lr_scheduler]

# --- Train ---
history = model.fit(
    X, Y,
    batch_size=16,
    epochs=100,
    validation_split=0.2,
    callbacks=cb,
    shuffle=True,
)

def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(15,5))

    # Loss
    plt.subplot(1,3,1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1,3,2)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Dice coefficient
    plt.subplot(1,3,3)
    plt.plot(epochs, history.history['dice_coefficient'], label='Train Dice')
    plt.plot(epochs, history.history['val_dice_coefficient'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)
model.save("final_wound_segmentation_model1.h5")
# Load the final saved model for testing
model = tf.keras.models.load_model(
    "final_wound_segmentation_model1.h5",
    custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient}
)

# --- Test Data Loading Functions ---
def load_test_image(path, target_size=(128, 128)):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0

def load_test_mask(path, target_size=(128, 128)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = mask / 255.0
    return np.expand_dims(mask, axis=-1)

test_image_dir = "E:/Pyhton for wound segmentation/wound-segmentation-master/wound-segmentation-master/data/wound_dataset/azh_wound_care_center_dataset_patches/test/images"
test_mask_dir = "E:/Pyhton for wound segmentation/wound-segmentation-master/wound-segmentation-master/data/wound_dataset/azh_wound_care_center_dataset_patches/test/labels"

test_image_list = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.png')])
test_mask_list = sorted([f for f in os.listdir(test_mask_dir) if f.endswith('.png')])

X_test, Y_test = [], []
for img_name in tqdm(test_image_list, desc="Loading test data"):
    img_path = os.path.join(test_image_dir, img_name)
    mask_path = os.path.join(test_mask_dir, img_name)
    if os.path.exists(img_path) and os.path.exists(mask_path):
        img = load_test_image(img_path)
        mask = load_test_mask(mask_path)
        if img is not None and mask is not None:
            X_test.append(img)
            Y_test.append(mask)

X_test = np.array(X_test, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)
print("Loaded test images shape:", X_test.shape)
print("Loaded test masks shape:", Y_test.shape)

pred_test = model.predict(X_test, batch_size=8)

pred_binary = (pred_test > 0.7).astype(np.uint8)
gt_binary = (Y_test > 0.7).astype(np.uint8)

y_true = gt_binary.flatten()
y_pred = pred_binary.flatten()

intersection = np.logical_and(y_true, y_pred).sum()
union = np.logical_or(y_true, y_pred).sum()
iou = (intersection + 1e-6) / (union + 1e-6)
dice = (2 * intersection + 1e-6) / (y_true.sum() + y_pred.sum() + 1e-6)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📊 Test Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")

test_metrics = [
    ["Metric", "Value"],
    ["Accuracy", acc],
    ["Precision", prec],
    ["Recall", rec],
    ["F1-Score", f1],
    ["IoU", iou],
    ["Dice", dice]
]

with open("test_metrics.csv", "w", newline="") as f:
    csv.writer(f).writerows(test_metrics)

print("Saved test metrics to test_metrics4.csv")