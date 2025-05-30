import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Flatten,
                                     Dropout, Multiply, GlobalAveragePooling2D, Reshape)

# Spatial Attention Layer
def spatial_attention_layer(x):
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, x.shape[-1]))(avg_pool)
    dense = Dense(x.shape[-1], activation='sigmoid')(avg_pool)
    return Multiply()([x, dense])

# CNN Model with Attention
def build_model(num_classes):
    inp = Input(shape=(64, 64, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = spatial_attention_layer(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)

# Load and preprocess data
data_dir = "Indian"
img_height = 64
img_width = 64
batch_size = 32

data_gen = ImageDataGenerator(rescale=1./255)
full_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=10000,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True
)

X, y = next(full_data)
for _ in range(1, len(full_data)):
    X_batch, y_batch = next(full_data)
    X = np.concatenate((X, X_batch))
    y = np.concatenate((y, y_batch))

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Build and compile model
num_classes = len(np.unique(y))
model = build_model(num_classes)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=batch_size,
    callbacks=[early_stop]
)

# Save the trained model
from keras.models import load_model
from keras.layers import InputLayer

model = load_model("indian_model.h5", custom_objects={"InputLayer": InputLayer})

# Evaluate
loss, acc = model.evaluate(X_val, y_val)
print(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")

# 📊 Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)
