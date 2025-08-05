import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = warning, 2 = error, 3 = none
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import sys
import os

# Load the model
model = load_model("model/gender_model.h5")

# Helper: preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Smart predictor
def smart_predict(image_path):
    try:
        # Extract info from filename
        filename = os.path.basename(image_path)
        name_parts = filename.split(".")[0].split("_")
        age = int(name_parts[0])
        gender = name_parts[1].lower()      # actual label, unused here
        hair = name_parts[2].lower()

        print(f"Age: {age}, Hair: {hair}")

        if 20 <= age <= 30:
            # Use hair length logic
            if hair == "long":
                predicted = "female"
            else:
                predicted = "male"
            logic_used = "Hair-based logic (age 20–30)"
        else:
            # Use model
            img = preprocess_image(image_path)
            pred = model.predict(img)[0]
            predicted = "female" if np.argmax(pred) == 1 else "male"
            logic_used = "CNN Model (age <20 or >30)"

        print(f"Predicted Gender: {predicted}  | Logic: {logic_used}\n")

    except Exception as e:
        print(f"Error: {e}")

# ------------ Run the script ------------
if __name__ == "__main__":
    test_image_path = input("Enter image path (like dataset/25_male_long.jpg): ")
    smart_predict(test_image_path)
