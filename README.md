
# 🍎 NutriVision AI
### AI-Powered Food Recognition & Nutrition Analysis

NutriVision AI is a machine learning-based web application designed to identify food items from uploaded images and provide approximate nutritional information. It combines computer vision, pretrained deep learning models, and an interactive web interface to make food analysis more accessible.

---

## ✨ Features

- **AI Food Recognition:** Identifies food items from uploaded images using a pretrained Food-101 classification model.
- **Top-3 Predictions:** Displays the three most likely food classifications with confidence scores.
- **Fruit & Vegetable Detection:** Uses a YOLO-based object detection model to identify supported fruits and vegetables.
- **Nutrition Estimation:** Provides approximate nutritional reference values for supported food categories.
- **Serving Size Adjustment:** Scales nutritional values according to the serving weight entered by the user.
- **Image Quality Analysis:** Displays basic image brightness and sharpness indicators.
- **Interactive Interface:** Supports image upload, drag-and-drop functionality, image preview, and removal.
- **Responsive Design:** Provides light and dark themes with a responsive layout.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python, JavaScript |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Machine Learning | PyTorch, Transformers |
| Computer Vision | YOLO, OpenCV |
| Model Hosting | Hugging Face |

---

## 🧠 Machine Learning Models

NutriVision AI integrates two pretrained models for food classification and object detection.

### 1. Food Classification

- **Model:** `nateraw/food`
- **Dataset:** Food-101
- **Architecture:** Vision Transformer (ViT)
- **Purpose:** Classifies food images and generates the top-3 predictions with confidence scores.

Model source: https://huggingface.co/nateraw/food

Dataset source: https://huggingface.co/datasets/ethz/food101

### 2. Fruit and Vegetable Detection

- **Model:** `Piyu12/fruit-veg-yolo11m-detector`
- **Architecture:** YOLO11m
- **Purpose:** Detects supported fruits and vegetables in uploaded images and identifies their locations.

Model source: https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector

The detector's training dataset and supported classes are described in its model documentation.

**Note:** Both models are pretrained models integrated into the application. They are not trained from scratch as part of this project.

---

## ⚙️ Algorithms and Methodology

NutriVision AI combines image classification, object detection, and reference-based nutrition estimation.

### 1. Food Image Classification

1. The user uploads a food image.
2. The image is preprocessed and passed to the pretrained Food-101 classification model.
3. The Vision Transformer analyzes image features and predicts the food category.
4. The application displays the top-3 predicted categories with their confidence scores.

### 2. Fruit and Vegetable Detection

1. The uploaded image is passed to the pretrained YOLO11m detection model.
2. The model identifies supported fruits and vegetables.
3. Detected objects are associated with their predicted labels and confidence scores.
4. Bounding boxes are used to indicate the detected objects in the image.

### 3. Nutrition Estimation

The application uses a predefined nutrition reference database containing approximate nutritional values per 100 grams.

The predicted food category is matched with the available nutrition data. The nutritional values are then adjusted according to the serving size entered by the user.

**Calculation:**

Estimated nutrient amount = (Nutrient value per 100 g × Serving size in grams) / 100

The application provides estimates for available nutritional information, such as calories, protein, carbohydrates, and fat.

### 4. Image Quality Analysis

Basic image processing techniques are used to calculate image brightness and sharpness indicators. These indicators help users understand the visual quality of the uploaded image.

---

## 🧪 Experimental Results and Model Evaluation

The project was evaluated through sample image testing to examine the functionality of food classification, fruit and vegetable detection, and nutrition estimation.

### Model Performance Comparison

| Model | Main Task | Evaluation Output |
|---|---|---|
| Food-101 Classification Model | Food image classification | Top-3 predictions and confidence scores |
| YOLO11m Detector | Fruit and vegetable detection | Object labels, confidence scores, and bounding boxes |

### Application Testing

The following functionalities are considered during application testing:

| Test Case | Expected Result |
|---|---|
| Upload a supported food image | Food category prediction |
| Analyze an image with multiple supported objects | Object detection results |
| View top-3 predictions | Three predicted categories with confidence scores |
| Change serving size | Updated nutritional estimates |
| Upload an unclear or unsupported image | Potentially incorrect or unsupported prediction |
| Change website theme | Light and dark themes work correctly |

**Evaluation Note:** The table describes the application's test cases and expected behaviour. Independent quantitative accuracy measurements for the complete application have not yet been established. Model confidence scores should not be treated as overall model accuracy.

---

## 📂 Project Structure

```text
AI_Food_Quality_Nutrition_Website/
│
├── server.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## 📚 Dataset and Model Sources

The project integrates pretrained models hosted on Hugging Face.

| Resource | Source |
|---|---|
| Food-101 Classification Model | https://huggingface.co/nateraw/food |
| Food-101 Dataset | https://huggingface.co/datasets/ethz/food101 |
| Fruit and Vegetable Detection Model | https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector |

The pretrained model weights are downloaded when required by the application. They are not included in the GitHub repository.

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Internet connection for downloading pretrained models

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kuheli18/NutriVision-AI.git
```

### Step 2: Open the Project

Open the cloned project folder in Visual Studio Code.

### Step 3: Install Dependencies

Open the terminal and execute:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python server.py
```

### Step 5: Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

The application will be available locally in your browser.

**Note:** The first model download may take several minutes depending on your internet connection.

---

## ⚠️ Limitations

- Food recognition may produce incorrect predictions for unfamiliar or visually similar foods.
- The application supports only the food categories covered by its integrated models and nutrition reference data.
- Nutritional values are approximate reference values, not measurements extracted directly from an image.
- Serving weight is entered manually by the user and is not estimated by the model.
- A photograph cannot reliably establish food freshness, spoilage, contamination, hidden ingredients, or food safety.
- Detection and nutrition estimation depend on the available model classes and reference data.
- Complex meals containing multiple ingredients may not receive a complete nutritional breakdown.

---

## 🔮 Future Improvements

- Expand the supported food and nutrition database.
- Improve recognition accuracy using a larger and more diverse dataset.
- Develop more comprehensive multi-food meal analysis.
- Improve nutritional estimation and ingredient identification.
- Add a dedicated quantitative model evaluation using a separate test dataset.
- Deploy the application on a public hosting platform.

---

## 🎓 Project Purpose

NutriVision AI is developed as an educational machine learning project to explore the practical integration of computer vision, pretrained deep learning models, and web development.

It demonstrates how AI-based image recognition can be combined with an interactive web application to provide basic food and nutrition insights.

---

**Disclaimer:** NutriVision AI is an educational prototype. It is not a certified nutrition, medical, or food-safety assessment tool.
