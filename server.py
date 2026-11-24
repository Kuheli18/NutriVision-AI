
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# AI models
food_classifier = None
produce_model = None

# Approximate nutrition values per 100 grams:
# calories, protein, carbs, fat, fiber
NUTRITION = {
    "apple": [52, 0.3, 13.8, 0.2, 2.4],
    "banana": [89, 1.1, 22.8, 0.3, 2.6],
    "orange": [47, 0.9, 11.8, 0.1, 2.4],
    "mango": [60, 0.8, 15, 0.4, 1.6],
    "grape": [69, 0.7, 18.1, 0.2, 0.9],
    "strawberry": [32, 0.7, 7.7, 0.3, 2],
    "pineapple": [50, 0.5, 13.1, 0.1, 1.4],
    "watermelon": [30, 0.6, 7.6, 0.2, 0.4],
    "pomegranate": [83, 1.7, 18.7, 1.2, 4],
    "pear": [57, 0.4, 15.2, 0.1, 3.1],
    "peach": [39, 0.9, 9.5, 0.3, 1.5],
    "kiwi": [61, 1.1, 14.7, 0.5, 3],
    "cherry": [63, 1.1, 16, 0.2, 2.1],
    "lemon": [29, 1.1, 9.3, 0.3, 2.8],
    "carrot": [41, 0.9, 9.6, 0.2, 2.8],
    "potato": [77, 2, 17.5, 0.1, 2.2],
    "tomato": [18, 0.9, 3.9, 0.2, 1.2],
    "onion": [40, 1.1, 9.3, 0.1, 1.7],
    "ginger": [80, 1.8, 17.8, 0.8, 2],
    "cucumber": [15, 0.7, 3.6, 0.1, 0.5],
    "cabbage": [25, 1.3, 5.8, 0.1, 2.5],
    "cauliflower": [25, 1.9, 5, 0.3, 2],
    "broccoli": [34, 2.8, 6.6, 0.4, 2.6],
    "beetroot": [43, 1.6, 9.6, 0.2, 2.8],
    "capsicum": [31, 1, 6, 0.3, 2.1],
    "peas": [81, 5.4, 14.5, 0.4, 5.1],
    "maize": [86, 3.3, 18.7, 1.4, 2],
    "mushroom": [22, 3.1, 3.3, 0.3, 1],
    "rice": [130, 2.7, 28, 0.3, 0.4],
    "bread": [265, 9, 49, 3.2, 2.7],
    "pizza": [266, 11, 33, 10, 2.3],
    "hamburger": [295, 17, 24, 14, 1.3],
    "cheeseburger": [300, 17, 30, 14, 1.5],
    "french fries": [312, 3.4, 41, 15, 3.8],
    "hot dog": [290, 10, 24, 18, 1],
    "fried rice": [174, 4.1, 25, 6, 1.2],
    "sushi": [150, 5, 30, 0.5, 1],
    "spaghetti": [158, 5.8, 30.9, 0.9, 1.8],
    "ice cream": [207, 3.5, 24, 11, 0.7],
    "cake": [350, 5, 50, 15, 1],
    "donut": [452, 4.9, 51, 25, 2],
    "omelette": [154, 10.6, 0.6, 11.7, 0],
    "egg": [155, 13, 1.1, 11, 0],
    "chicken": [165, 31, 0, 3.6, 0],
    "salad": [33, 1.5, 6, 0.4, 2.5],
    "sandwich": [250, 10, 30, 10, 2],
    "pasta": [158, 5.8, 30.9, 0.9, 1.8],
    "soup": [45, 2, 6, 1.5, 1],
    "steak": [271, 25, 0, 19, 0],
    "guacamole": [150, 2, 8.5, 13, 6.7]
}

NUTRIENT_NAMES = ["calories", "protein", "carbs", "fat", "fiber"]


def get_food_classifier():
    global food_classifier

    if food_classifier is None:
        print("Loading Food-101 model...")
        food_classifier = pipeline(
            "image-classification",
            model="nateraw/food"
        )

    return food_classifier


def get_produce_model():
    global produce_model

    if produce_model is None:
        print("Loading fruit and vegetable detection model...")

        model_path = hf_hub_download(
            repo_id="Piyu12/fruit-veg-yolo11m-detector",
            filename="best.pt"
        )

        produce_model = YOLO(model_path)

    return produce_model



def nutrition_for(label):
    label = label.lower().replace("_", " ").strip()

    # Check individual food names in combined predictions
    for food, values in NUTRITION.items():
        if food in label:
            return food, dict(zip(NUTRIENT_NAMES, values))

    return None, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "food_image" not in request.files:
        return jsonify(error="Please select an image first."), 400

    file = request.files["food_image"]

    if file.filename == "":
        return jsonify(error="Please select an image first."), 400

    # Read image
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify(
            error="Invalid image file. Please upload JPG, PNG or WEBP."
        ), 400


    try:
        # Load both AI models
        model = get_produce_model()
        classifier = get_food_classifier()

        # Fruit and vegetable detection
        results = model.predict(
            source=image,
            conf=0.30,
            imgsz=640,
            verbose=False
        )

        detected = {}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                detected_label = model.names[class_id]
                detected_label = detected_label.replace("_", " ").lower()

                if detected_label not in detected:
                    detected[detected_label] = {
                        "label": detected_label,
                        "score": confidence,
                        "count": 1
                    }
                else:
                    detected[detected_label]["count"] += 1

                    if confidence > detected[detected_label]["score"]:
                        detected[detected_label]["score"] = confidence

        # Sort produce detections by confidence
        produce_predictions = sorted(
            detected.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Prepared food classification
        food_predictions = classifier(image, top_k=3)

        food_label = (
            food_predictions[0]["label"]
            .lower()
            .replace("_", " ")
        )
        food_score = float(food_predictions[0]["score"])

        # Common prepared-food categories
        prepared_foods = {
            "spaghetti", "spaghetti bolognese",
            "spaghetti carbonara", "pizza",
            "hamburger", "cheeseburger", "hot dog",
            "french fries", "fried rice", "sushi",
            "ice cream", "cake", "donut", "omelette",
            "sandwich", "pasta", "soup", "steak",
            "guacamole", "chicken", "lasagna",
            "ramen", "dumplings", "spring rolls",
            "risotto", "tacos"
        }

        # Select the appropriate prediction
        if food_label in prepared_foods and food_score >= 0.20:

            label = food_label

            predictions = [
                {
                    "label": item["label"].replace("_", " ").title(),
                    "score": item["score"] * 100
                }
                for item in food_predictions
            ]

            detected_items = []
            detection_type = "Prepared Food Classification"

        elif produce_predictions:

            label = produce_predictions[0]["label"]

            predictions = [
                {
                    "label": item["label"].title(),
                    "score": item["score"] * 100
                }
                for item in produce_predictions[:5]
            ]

            detected_items = [
                {
                    "label": item["label"].title(),
                    "count": item["count"],
                    "score": round(item["score"] * 100, 1)
                }
                for item in produce_predictions
            ]

            detection_type = "Fruit and Vegetable Detection"

        else:

            label = food_label

            predictions = [
                {
                    "label": item["label"].replace("_", " ").title(),
                    "score": item["score"] * 100
                }
                for item in food_predictions
            ]

            detected_items = []
            detection_type = "Food Classification"

    except Exception as exc:
        app.logger.exception("AI model error")

        return jsonify(
            error="Could not run the AI model. Check your internet connection.",
            details=str(exc)[:300]
        ), 500


    # Nutrition information for the primary prediction
    food, nutrition = nutrition_for(label)

    # Image quality analysis
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    quality = {
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "lighting": (
            "Very dark" if brightness < 45
            else "Very bright" if brightness > 220
            else "Moderate"
        ),
        "focus": (
            "Possibly blurry" if sharpness < 40
            else "Reasonably sharp"
        )
    }

    return jsonify(
        food_label=label.replace("_", " ").title(),
        score=round(float(predictions[0]["score"]), 1),
        top_predictions=[
            {
                "label": p["label"].replace("_", " ").title(),
                "score": round(float(p["score"]), 1)
            }
            for p in predictions
        ],
        detected_items=detected_items,
        detection_type=detection_type,
        nutrition_name=food.title() if food else None,
        nutrition_per_100g=nutrition,
        image_quality=quality
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        error="Image too large. Maximum size is 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
