# NutriVision AI — Flask Website

## Run on Windows / VS Code
1. Extract this ZIP.
2. In VS Code choose **File → Open Folder** and select the extracted `AI_Food_Quality_Nutrition_Website` folder.
3. Open **Terminal → New Terminal**. Run `dir` and confirm `server.py` and `requirements.txt` appear.
4. Install packages: `python -m pip install -r requirements.txt`
5. Start the website: `python server.py`
6. Open `http://127.0.0.1:5000` in your browser.

The first image analysis downloads the pretrained `nateraw/food` classifier from Hugging Face. Internet access is needed and the first download may take a few minutes.

## Features
- Food image classification with a pretrained Food-101 model.
- Top three predicted labels and model scores.
- Approximate nutrition lookup for a limited list of foods.
- Serving-size scaling based on the user's selected gram amount.
- Basic image brightness and sharpness indicators.
- Responsive design and light/dark theme.

## Limitations
This is an educational prototype, not a certified nutrition or food-safety tool. Food recognition can be wrong. Nutrition values are approximate reference values, not measured from the image. Serving weight is selected by the user, not inferred by the model. A photo cannot reliably determine freshness, spoilage, contamination, hidden ingredients or food safety. For academic evaluation, use a separate labelled test set and report accuracy, precision, recall and F1-score. Do not claim this project trained the pretrained model from scratch.
