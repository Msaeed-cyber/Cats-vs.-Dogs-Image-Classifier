import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Settings
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 5
DATASET_PATH = "C:/Users/lenovo/Desktop/cats_dog/cats_dogs_light/cats_dogs_light/train"  # <- update this if your path is different

# -------------------------
# 1. Load and Prepare Dataset
# -------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True, zoom_range=0.2)

train_data = datagen.flow_from_directory(DATASET_PATH,
                                         target_size=(IMG_SIZE, IMG_SIZE),
                                         batch_size=BATCH_SIZE,
                                         class_mode='binary',
                                         subset='training')

val_data = datagen.flow_from_directory(DATASET_PATH,
                                       target_size=(IMG_SIZE, IMG_SIZE),
                                       batch_size=BATCH_SIZE,
                                       class_mode='binary',
                                       subset='validation')

# -------------------------
# 2. Load Pretrained Model
# -------------------------
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Freeze base layers

# -------------------------
# 3. Build Model
# -------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------------
# 4. Train the Model
# -------------------------
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=EPOCHS)

# -------------------------
# 5. Save the Trained Model
# -------------------------
model.save("cats_dogs_mobilenetv2.h5")
print("✅ Model saved as 'cats_dogs_mobilenetv2.h5'")

# -------------------------
# 6. Load & Predict Image from PC
# -------------------------
# Load the saved model again (optional)
model = tf.keras.models.load_model("cats_dogs_mobilenetv2.h5")

def load_and_predict():
    # Open file dialog to choose image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        print("No file selected.")
        return

    # Load and preprocess image
    img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"\n📷 Prediction: {label} | Confidence: {confidence:.2f}")

    # Display image with prediction
    plt.imshow(img)
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()

# Open file picker GUI
root = tk.Tk()
root.withdraw()
load_and_predict()
