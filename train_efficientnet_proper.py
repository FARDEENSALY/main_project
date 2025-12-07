import os
import math
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications.efficientnet import preprocess_input

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded successfully!')

# Define data paths
data_dir = 'pets_facial_expression_dataset'
ds_name = 'Pets Facial Expression'

# Generate data paths with labels
def generate_data_paths(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)
    for fold in folds:
        if fold == 'master':
            continue
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return filepaths, labels

# Create dataframe
def create_df(filepaths, labels):
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

# Generate data and create dataframe
filepaths, labels = generate_data_paths(data_dir)
df = create_df(filepaths, labels)

print(f"Dataset loaded: {len(df)} images")
print(f"Classes: {df['labels'].unique()}")

# Split dataframe into train, valid, and test
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

print(f"Training: {len(train_df)} images")
print(f"Validation: {len(valid_df)} images")
print(f"Testing: {len(test_df)} images")

# Create Image Data Generator
batch_size = 8
img_size = (300, 300)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Data generators with proper preprocessing for EfficientNet
tr_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                           rotation_range=20,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.1,
                           brightness_range=[0.8,1.2],
                           zoom_range=0.1,
                           horizontal_flip=True,
                           fill_mode='nearest')

ts_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Calculate test batch size
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

print(f"Test batch size: {test_batch_size}, Test steps: {test_steps}")

# Create generators
train_gen = tr_gen.flow_from_dataframe(train_df,
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=True,
                                       batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df,
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=True,
                                       batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df,
                                      x_col='filepaths',
                                      y_col='labels',
                                      target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb',
                                      shuffle=False,
                                      batch_size=test_batch_size)

# Model Structure
class_count = len(list(train_gen.class_indices.keys()))
print(f"Number of classes: {class_count}")
print(f"Class indices: {train_gen.class_indices}")

# Create pre-trained model
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,
                                                               weights="imagenet",
                                                               input_shape=img_shape,
                                                               pooling='max')
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, activation='relu'),
    Dense(128, kernel_regularizer=regularizers.l2(l2=0.016),  # Fixed: l2 instead of l
                activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006),
                activation='relu'),
    Dropout(rate=0.3, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Summary:")
model.summary()

# Define Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=15,
                               restore_best_weights=True,
                               mode='max')

# Learning rate scheduler for phase 1
def lr_schedule_phase1(epoch):
    lr = 0.001
    if epoch > 15:
        lr *= 0.1
    return lr

lr_scheduler_phase1 = LearningRateScheduler(lr_schedule_phase1)

# Learning rate scheduler for fine-tuning (phase 2)
def lr_schedule_phase2(epoch):
    lr = 1e-5
    if epoch > 20:
        lr *= 0.1
    return lr

lr_scheduler_phase2 = LearningRateScheduler(lr_schedule_phase2)

# Learning rate scheduler for full fine-tuning (phase 3)
def lr_schedule_phase3(epoch):
    lr = 1e-6
    if epoch > 10:
        lr *= 0.1
    return lr

lr_scheduler_phase3 = LearningRateScheduler(lr_schedule_phase3)

callbacks_phase1 = [early_stopping, lr_scheduler_phase1]
callbacks_phase2 = [early_stopping, lr_scheduler_phase2]
callbacks_phase3 = [early_stopping, lr_scheduler_phase3]

# Three-phase training: Phase 1 - Train top layers, Phase 2 - Fine-tune base model, Phase 3 - Full fine-tuning

# Phase 1: Train top layers with frozen base model
print("🚀 Phase 1: Training top layers (30 epochs)...")
print("Base model is frozen, training only the classification head...")

epochs_phase1 = 30
history_phase1 = model.fit(x=train_gen,
                          epochs=epochs_phase1,
                          verbose=1,
                          validation_data=valid_gen,
                          validation_steps=None,
                          shuffle=False,
                          callbacks=callbacks_phase1)

print("✅ Phase 1 completed!")

# Phase 2: Fine-tune by unfreezing some base model layers
print("\n🔧 Phase 2: Fine-tuning base model layers...")
print("Unfreezing top layers of EfficientNetB5 for fine-tuning...")

# Unfreeze the last 50 layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(Adamax(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model recompiled for fine-tuning with lower learning rate.")

epochs_phase2 = 70  # Total remaining epochs
history_phase2 = model.fit(x=train_gen,
                          epochs=epochs_phase2,
                          verbose=1,
                          validation_data=valid_gen,
                          validation_steps=None,
                          shuffle=False,
                          callbacks=callbacks_phase2)

print("✅ Phase 2 (fine-tuning) completed!")

# Phase 3: Full fine-tuning of all layers
print("\n🔧 Phase 3: Full fine-tuning of all model layers...")
print("Unfreezing all base model layers for complete fine-tuning...")

# Unfreeze all layers for full fine-tuning
base_model.trainable = True

# Recompile with very low learning rate for full fine-tuning
model.compile(Adamax(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model recompiled for full fine-tuning with very low learning rate.")

epochs_phase3 = 20  # Limited epochs to avoid overfitting
history_phase3 = model.fit(x=train_gen,
                          epochs=epochs_phase3,
                          verbose=1,
                          validation_data=valid_gen,
                          validation_steps=None,
                          shuffle=False,
                          callbacks=callbacks_phase3)

print("✅ Phase 3 (full fine-tuning) completed!")

# Combine histories for plotting
def combine_histories(hist1, hist2, hist3):
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key] + hist3.history[key]
    return combined

history = combine_histories(history_phase1, history_phase2, history_phase3)

print("✅ Three-phase training completed!")

# Evaluate Model
print("\n📊 Evaluating model performance...")

# Calculate scores
train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print("\n" + "="*50)
print("FINAL RESULTS:")
print("="*50)
print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print("-" * 30)
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print("-" * 30)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")
print("="*50)

# Get Predictions for detailed analysis
print("\n🔍 Generating detailed predictions...")
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)

# Confusion Matrix
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

cm = confusion_matrix(test_gen.classes, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(test_gen.classes, y_pred, target_names=classes))

# Plot training history
tr_acc = history['accuracy']
tr_loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]

plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=f'best epoch= {str(index_loss + 1)}')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=f'best epoch= {str(index_acc + 1)}')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("📈 Training history plot saved as 'training_history.png'")

# Save the Model
model.save('efficientnet_pet_emotion_proper.h5')
print("💾 Model saved as 'efficientnet_pet_emotion_proper.h5'")

print("\n🎉 Training and evaluation completed!")
print(f"🎯 Achieved Test Accuracy: {test_score[1]*100:.2f}%")
print("📝 This should match the original notebook's 93-94% accuracy with proper training.")
