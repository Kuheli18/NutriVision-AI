# 🍎 NutriVision AI
### AI-Powered Food Recognition & Nutrition Analysis

NutriVision AI is a machine learning-based web application designed to identify food items from uploaded images and provide approximate nutritional information. It combines computer vision and a simple, interactive web interface to make food analysis more accessible.

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

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python, JavaScript |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Machine Learning | PyTorch, Transformers |
| Computer Vision | YOLO, OpenCV |
| Model Hosting | Hugging Face |

## 🧠 Machine Learning Models

**1. Food Classification**

- Model: `nateraw/food`
- Based on the Food-101 dataset.
- Used to classify food images and generate the top predictions.

**2. Fruit and Vegetable Detection**

- Model: `Piyu12/fruit-veg-yolo11m-detector`
- Used to detect supported fruits and vegetables in uploaded images.

Both models are pretrained models integrated into the application.

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

## ⚠️ Limitations

- Food recognition may produce incorrect predictions for unfamiliar or visually similar foods.
- The application supports only the food categories covered by its integrated models and nutrition reference data.
- Nutritional values are approximate reference values, not measurements extracted directly from an image.
- Serving weight is entered manually by the user and is not estimated by the model.
- A photograph cannot reliably establish food freshness, spoilage, contamination, hidden ingredients, or food safety.
- Detection and nutrition estimation depend on the available model classes and reference data.

## 🔮 Future Improvements

- Expand the supported food and nutrition database.
- Improve recognition accuracy using a larger and more diverse dataset.
- Develop more comprehensive multi-food meal analysis.
- Improve nutritional estimation and ingredient identification.
- Deploy the application on a public hosting platform.

## 🎓 Project Purpose

NutriVision AI is developed as an educational machine learning project to explore the practical integration of computer vision, pretrained deep learning models, and web development.

It demonstrates how AI-based image recognition can be combined with an interactive web application to provide basic food and nutrition insights.

---

**Disclaimer:** NutriVision AI is an educational prototype. It is not a certified nutrition, medical, or food-safety assessment tool.
