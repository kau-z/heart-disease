# ❤️ Heart Disease Risk Predictor

A Streamlit web app that predicts the risk of heart disease using a trained Random Forest model and explains key factors influencing the prediction with SHAP.

## ✨ Features
- **Risk Prediction**: Enter age, cholesterol, blood pressure, and other clinical indicators to estimate heart-disease risk.
- **Feature Importance**: Interactive SHAP explanations show which factors increase or decrease risk for each prediction.
- **Personalized Wellness Tips**: Automatically generated health suggestions based on your inputs and model insights.
- **History Tracking**: Saves each prediction so you can review or delete past results.


## 📂 Project Structure
├─ data/
│ └─ Heart-Disease.csv # Original dataset
├─ notebooks/
│ ├─ 01_EDA_Modeling.ipynb # EDA and base modeling
│ ├─ 02_Model_Tuning.ipynb # Hyper-parameter tuning & evaluation
│ └─ outputs/ # Model artifacts & figures
│ ├─ rf_model.joblib
│ ├─ columns.joblib
│ └─ cleaned_heart.csv
├─ src/
│ └─ app.py # Streamlit app
├─ .gitignore
├─ README.md
└─ requirements.txt


---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kau-z/heart-disease.git
cd heart-disease

2️⃣ Install dependencies

Create and activate a virtual environment (recommended) and install requirements:

pip install -r requirements.txt

3️⃣ Run the app
streamlit run src/app.py


Then open the link printed in your terminal (usually http://localhost:8501).

🧮 Model

Algorithm: Random Forest Classifier

Training Data: Cleaned Kaggle Heart Disease dataset

Explanation: SHAP (SHapley Additive exPlanations) to show per-feature impact.

⚠️ Disclaimer

This app is for educational and informational purposes only.
It does not provide medical advice and should not replace professional healthcare consultation.

🤝 Contributing

Pull requests and feature ideas are welcome!
For major changes, open an issue first to discuss what you’d like to add.