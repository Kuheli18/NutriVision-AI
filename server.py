import os
import io
import joblib
import numpy as np
import cv2
import torch
import torchvision.models as models

from PIL import Image
from flask import Flask, request, jsonify
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import normalize

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load trained SVM and food categories
svm_model = joblib.load(os.path.join(MODEL_DIR, "best_svm.pkl"))
food_classes = joblib.load(os.path.join(MODEL_DIR, "food_classes.pkl"))
# Load experimental classifiers
experimental_classifiers = {
    "SVM": svm_model,
    "KNN": joblib.load(os.path.join(MODEL_DIR, "improved_knn.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "improved_decision_tree.pkl"))
}
# Load the same ResNet50 feature extractor used during training
device = torch.device("cpu")

weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
).to(device)

feature_extractor.eval()
transform = weights.transforms()

# Approximate nutrition values per 100 grams.
# Values are estimates, not laboratory measurements.
NUTRITION = {
    "apple": [52, 0.3, 13.8, 0.2, 2.4],
    "banana": [89, 1.1, 22.8, 0.3, 2.6],
    "orange": [47, 0.9, 11.8, 0.1, 2.4],
    "broccoli": [34, 2.8, 6.6, 0.4, 2.6],
    "carrot": [41, 0.9, 9.6, 0.2, 2.8],
    "pizza": [266, 11, 33, 10, 2.3],
    "hamburger": [295, 17, 24, 14, 1.3],
    "cheeseburger": [303, 17, 30, 14, 1.3],
    "french_fries": [312, 3.4, 41, 15, 3.8],
    "hot_dog": [290, 10, 24, 18, 1.2],
    "fried_rice": [174, 5, 25, 6, 1.5],
    "sushi": [150, 6, 30, 1, 1],
    "spaghetti": [158, 5.8, 31, 0.9, 1.8],
    "ice_cream": [207, 3.5, 24, 11, 0.7],
    "cake": [350, 5, 50, 15, 1],
    "donuts": [452, 5, 51, 25, 1.7],
    "omelette": [154, 11, 1.5, 11, 0],
    "chicken_wings": [203, 30, 0, 8.1, 0],
    "salad": [20, 1.5, 3.5, 0.2, 1.5],
    "tomato": [18, 0.9, 3.9, 0.2, 1.2],
    "bread": [265, 9, 49, 3.2, 2.7],
    "sandwich": [250, 10, 30, 10, 2],
    "pasta": [158, 5.8, 31, 0.9, 1.8],
    "soup": [50, 2, 7, 1.5, 1],
    "steak": [271, 25, 0, 19, 0],
    "guacamole": [150, 2, 9, 13, 7],
}

NUTRITION_FIELDS = [
    "calories", "protein", "carbs", "fat", "fibre"
]


def extract_features(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(tensor)

    features = features.squeeze().cpu().numpy().reshape(1, -1)

    # Must match the L2 normalization used during training
    return normalize(features, norm="l2")


def get_nutrition(label):

    label = label.lower().strip()

    # Exact match
    if label in NUTRITION:
        values = NUTRITION[label]

    # Spaghetti variations
    elif label.startswith("spaghetti"):
        values = NUTRITION["spaghetti"]

    # Pasta variations
    elif label.startswith("pasta"):
        values = NUTRITION["pasta"]

    # Soup variations
    elif label.endswith("soup") or label == "soup":
        values = NUTRITION["soup"]

    else:
        return None

    return dict(zip(NUTRITION_FIELDS, values))


def get_image_quality(image):
    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(
        gray, cv2.CV_64F
    ).var())

    lighting = (
        "Good" if 70 <= brightness <= 200
        else "Too dark" if brightness < 70
        else "Too bright"
    )

    focus = "Good" if sharpness >= 100 else "Low"

    return {
        "lighting": lighting,
        "focus": focus,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2)
    }
def get_experimental_predictions(features):
    predictions = []
    labels = []

    for model_name, model in experimental_classifiers.items():
        try:
            predicted_id = int(model.predict(features)[0])
            label = food_classes[predicted_id]

            labels.append(label)

            predictions.append({
                "model": model_name,
                "prediction": label.replace("_", " ").title()
            })

        except Exception as error:
            print(model_name, "prediction error:", error)

    if not predictions:
        return {
            "predictions": [],
            "error": "Unable to generate experimental predictions."
        }

    from collections import Counter

    counts = Counter(labels)
    majority_label, agreement = counts.most_common(1)[0]

    return {
        "predictions": predictions,
        "majority_prediction": majority_label.replace("_", " ").title(),
        "agreement": agreement,
        "total_models": len(predictions)
    }

@app.route("/")
def home():
    return app.send_static_file("index.html") if False else \
        __import__("flask").render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "food_image" not in request.files:
            return jsonify({"error": "Please upload a food image."}), 400

        file = request.files["food_image"]

        if not file.filename:
            return jsonify({"error": "No image selected."}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        features = extract_features(image)

        probabilities = svm_model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = []

        for index in top_indices:
            class_id = int(svm_model.classes_[index])
            label = food_classes[class_id]

            top_predictions.append({
                "label": label.replace("_", " ").title(),
                "score": round(float(probabilities[index]) * 100, 2)
            })

        predicted_id = int(svm_model.predict(features)[0])
        food_label = food_classes[predicted_id]

        nutrition = get_nutrition(food_label)
        experimental_results = get_experimental_predictions(features)

        return jsonify({
            "food_label": food_label.replace("_", " ").title(),
            "score": round(float(np.max(probabilities)) * 100, 2),
            "top_predictions": top_predictions,
            "nutrition_name": food_label.replace("_", " ").title(),
            "nutrition_per_100g": nutrition,
            "nutrition_available": nutrition is not None,
            "image_quality": get_image_quality(image),
            "experimental_models": experimental_results
        })

    except Exception as error:
        print("Analysis error:", str(error))
        return jsonify({
            "error": "Unable to analyze this image. Please try another image."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
