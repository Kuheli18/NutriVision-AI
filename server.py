import os
import io
from collections import Counter

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import normalize
from torchvision import transforms

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load the fine-tuned ResNet34 model (84.27% Food-101 test accuracy)
# ---------------------------------------------------------------------------
PT_MODEL_PATH = os.path.join(MODEL_DIR, "best_nutrivision_model.pt")
checkpoint = torch.load(PT_MODEL_PATH, map_location=device)
food_classes = list(checkpoint["classes"])

finetuned_model = models.resnet34(weights=None)
in_features = finetuned_model.fc.in_features
finetuned_model.fc = nn.Linear(in_features, len(food_classes))
finetuned_model.load_state_dict(checkpoint["model_state_dict"])
finetuned_model = finetuned_model.to(device)
finetuned_model.eval()

# Exact preprocessing used during test evaluation in Kaggle
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# 2. Optional legacy classical models (kept so frontend comparison UI works)
# ---------------------------------------------------------------------------
experimental_classifiers = {}
legacy_feature_extractor = None
legacy_transform = None

legacy_files = {
    "SVM": os.path.join(MODEL_DIR, "best_svm.pkl"),
    "KNN": os.path.join(MODEL_DIR, "improved_knn.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "improved_decision_tree.pkl"),
}

if any(os.path.exists(path) for path in legacy_files.values()):
    for name, path in legacy_files.items():
        if os.path.exists(path):
            experimental_classifiers[name] = joblib.load(path)

    weights = models.ResNet50_Weights.DEFAULT
    base_resnet = models.resnet50(weights=weights)
    legacy_feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1]).to(device)
    legacy_feature_extractor.eval()
    legacy_transform = weights.transforms()

# ---------------------------------------------------------------------------
# 3. Approximate nutrition values per 100 grams [calories, protein, carbs, fat, fibre]
# ---------------------------------------------------------------------------
NUTRITION = {
    "apple": [52, 0.3, 13.8, 0.2, 2.4],
    "apple_pie": [237, 2.4, 34.0, 11.0, 1.6],
    "baby_back_ribs": [292, 23.0, 6.0, 19.0, 0.3],
    "baklava": [428, 6.7, 51.0, 23.0, 2.5],
    "banana": [89, 1.1, 22.8, 0.3, 2.6],
    "beef_carpaccio": [135, 21.0, 1.0, 5.0, 0.0],
    "beef_tartare": [196, 20.0, 1.5, 12.0, 0.2],
    "beet_salad": [75, 2.2, 9.5, 3.5, 2.6],
    "beignets": [345, 5.5, 44.0, 16.0, 1.5],
    "bibimbap": [145, 6.5, 21.0, 3.8, 2.1],
    "bread": [265, 9.0, 49.0, 3.2, 2.7],
    "bread_pudding": [212, 5.4, 31.0, 7.2, 1.1],
    "breakfast_burrito": [215, 10.5, 21.0, 9.8, 2.0],
    "broccoli": [34, 2.8, 6.6, 0.4, 2.6],
    "caesar_salad": [180, 7.0, 8.0, 14.0, 1.8],
    "cake": [350, 5.0, 50.0, 15.0, 1.0],
    "carrot": [41, 0.9, 9.6, 0.2, 2.8],
    "cheeseburger": [303, 17.0, 30.0, 14.0, 1.3],
    "chicken_wings": [203, 30.0, 0.0, 8.1, 0.0],
    "chocolate_cake": [371, 5.0, 53.0, 16.0, 2.3],
    "donuts": [452, 5.0, 51.0, 25.0, 1.7],
    "french_fries": [312, 3.4, 41.0, 15.0, 3.8],
    "french_toast": [229, 7.7, 25.0, 10.8, 1.3],
    "fried_rice": [174, 5.0, 25.0, 6.0, 1.5],
    "guacamole": [150, 2.0, 9.0, 13.0, 7.0],
    "hamburger": [295, 17.0, 24.0, 14.0, 1.3],
    "hot_dog": [290, 10.0, 24.0, 18.0, 1.2],
    "ice_cream": [207, 3.5, 24.0, 11.0, 0.7],
    "macaroni_and_cheese": [164, 6.4, 18.0, 7.3, 1.1],
    "omelette": [154, 11.0, 1.5, 11.0, 0.0],
    "orange": [47, 0.9, 11.8, 0.1, 2.4],
    "pad_thai": [165, 7.5, 22.0, 5.5, 1.6],
    "pasta": [158, 5.8, 31.0, 0.9, 1.8],
    "pizza": [266, 11.0, 33.0, 10.0, 2.3],
    "ramen": [138, 6.2, 19.0, 4.1, 1.2],
    "salad": [20, 1.5, 3.5, 0.2, 1.5],
    "sandwich": [250, 10.0, 30.0, 10.0, 2.0],
    "soup": [50, 2.0, 7.0, 1.5, 1.0],
    "spaghetti": [158, 5.8, 31.0, 0.9, 1.8],
    "steak": [271, 25.0, 0.0, 19.0, 0.0],
    "sushi": [150, 6.0, 30.0, 1.0, 1.0],
    "tacos": [226, 9.4, 20.0, 12.0, 2.8],
    "tomato": [18, 0.9, 3.9, 0.2, 1.2],
}

NUTRITION_FIELDS = ["calories", "protein", "carbs", "fat", "fibre"]


def predict_finetuned(image):
    tensor = test_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = finetuned_model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "label": food_classes[int(idx)].replace("_", " ").title(),
            "score": round(float(probabilities[idx]) * 100, 2),
        }
        for idx in top_indices
    ]

    best_idx = int(top_indices[0])
    best_label = food_classes[best_idx]
    best_score = round(float(probabilities[best_idx]) * 100, 2)

    return best_label, best_score, top_predictions


def get_nutrition(label):
    label = label.lower().strip()

    if label in NUTRITION:
        values = NUTRITION[label]
    elif label.startswith("spaghetti"):
        values = NUTRITION["spaghetti"]
    elif label.startswith("pasta"):
        values = NUTRITION["pasta"]
    elif label.endswith("soup") or label == "soup":
        values = NUTRITION["soup"]
    elif label.endswith("salad"):
        values = NUTRITION["salad"]
    elif label.endswith("cake"):
        values = NUTRITION["cake"]
    elif label.endswith("sandwich"):
        values = NUTRITION["sandwich"]
    else:
        return None

    return dict(zip(NUTRITION_FIELDS, values))


def get_image_quality(image):
    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

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
        "sharpness": round(sharpness, 2),
    }


def get_experimental_predictions(image, primary_label):
    predictions = [
        {
            "model": "ResNet34 (Fine-Tuned)",
            "prediction": primary_label.replace("_", " ").title(),
        }
    ]
    labels = [primary_label]

    if experimental_classifiers and legacy_feature_extractor is not None:
        tensor = legacy_transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = legacy_feature_extractor(tensor)
        features = normalize(features.squeeze().cpu().numpy().reshape(1, -1), norm="l2")

        for model_name, model in experimental_classifiers.items():
            try:
                predicted_id = int(model.predict(features)[0])
                label = food_classes[predicted_id]
                labels.append(label)
                predictions.append({
                    "model": model_name,
                    "prediction": label.replace("_", " ").title(),
                })
            except Exception as error:
                print(model_name, "prediction error:", error)

    counts = Counter(labels)
    majority_label, agreement = counts.most_common(1)[0]

    return {
        "predictions": predictions,
        "majority_prediction": majority_label.replace("_", " ").title(),
        "agreement": agreement,
        "total_models": len(predictions),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "food_image" not in request.files:
            return jsonify({"error": "Please upload a food image."}), 400

        file = request.files["food_image"]
        if not file.filename:
            return jsonify({"error": "No image selected."}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        food_label, confidence_score, top_predictions = predict_finetuned(image)
        nutrition = get_nutrition(food_label)
        experimental_results = get_experimental_predictions(image, food_label)

        return jsonify({
            "food_label": food_label.replace("_", " ").title(),
            "score": confidence_score,
            "top_predictions": top_predictions,
            "nutrition_name": food_label.replace("_", " ").title(),
            "nutrition_per_100g": nutrition,
            "nutrition_available": nutrition is not None,
            "image_quality": get_image_quality(image),
            "experimental_models": experimental_results,
        })

    except Exception as error:
        print("Analysis error:", str(error))
        return jsonify({
            "error": "Unable to analyze this image. Please try another image."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)