import os
from flask import Flask, render_template, request
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = warning, 2 = error, 3 = none
from tensorflow.keras.models import load_model# pyright: ignore[reportMissingImports]

# App setup
app = Flask(__name__)
UPLOAD_FOLDER = "gui/static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("model/gender_model.h5")

# Preprocess function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Smart prediction logic
def smart_predict(image_path):
    try:
        filename = os.path.basename(image_path)
        name_parts = filename.split(".")[0].split("_")
        age = int(name_parts[0])
        hair = name_parts[2].lower()

        if 20 <= age <= 30:
            predicted = "Female" if hair == "long" else "Male"
            logic = "Hair-based logic (age 20–30)"
        else:
            img = preprocess_image(image_path)
            pred = model.predict(img)[0]
            predicted = "Female" if np.argmax(pred) == 1 else "Male"
            logic = "Model prediction (age <20 or >30)"

        return predicted, logic

    except Exception as e:
        return "Error", str(e)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = logic_used = filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            predicted, logic = smart_predict(path)
            prediction = predicted
            logic_used = logic

    return render_template("index.html", prediction=prediction, logic=logic_used, filename=filename)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
