import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping # pyright: ignore[reportMissingImports]
import joblib

# Load CSV
df = pd.read_csv("processed.csv")

# Prepare data
images = []
labels = []

for _, row in df.iterrows():
    img = cv2.imread(row['image'])
    if img is not None:
        img = cv2.resize(img, (128, 128))  # Resize all images
        img = img / 255.0  # Normalize
        images.append(img)

        # Convert gender to binary label
        label = 0 if row['gender'] == 'male' else 1
        labels.append(label)

X = np.array(images)
y = to_categorical(labels, num_classes=2)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save the model
model.save("model/gender_model.h5")

print("Model training complete and saved to model/gender_model.h5")
