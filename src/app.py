import os
import datetime
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Load model, scaler, columns, and training data
# ------------------------------------------------------
base = os.path.dirname(os.path.dirname(__file__))  # project root
rf      = joblib.load(os.path.join(base, "notebooks", "outputs", "rf_model.joblib"))
scaler  = joblib.load(os.path.join(base, "notebooks", "outputs", "scaler.joblib"))
columns = joblib.load(os.path.join(base, "notebooks", "outputs", "columns.joblib"))
train_df = pd.read_csv(os.path.join(base, "notebooks", "outputs", "cleaned_heart.csv"))

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Predicts heart-disease risk and highlights key contributing factors. "
           "This is **not** a medical diagnosis.")

# ------------------------------------------------------
# Collect user inputs
# ------------------------------------------------------
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"]
)
max_hr = st.number_input("Maximum Heart Rate", 60, 210, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

history_file = os.path.join(base, "user_history.csv")

# ------------------------------------------------------
# Prediction + Explanations
# ------------------------------------------------------
if st.button("Predict"):
    record = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain,
        "resting_blood_pressure": resting_bp,
        "cholestoral": cholesterol,          # match CSV spelling
        "fasting_blood_sugar": fasting_bs,
        "rest_ecg": resting_ecg,
        "Max_heart_rate": max_hr,
        "exercise_induced_angina": exercise_angina,
        "oldpeak": oldpeak,
        "slope": st_slope
    }

    # Encode & scale
    df = pd.DataFrame([record])
    df_enc = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(df_enc)

    # Predict risk
    prob = rf.predict_proba(X_scaled)[0, 1]
    st.subheader(f"Predicted Risk of Heart Disease: **{prob*100:.1f}%**")

    # ---------- SHAP explanation ----------
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        sample_shap = shap_values[1][0]
    elif getattr(shap_values, "ndim", 2) == 3:
        sample_shap = shap_values[0, :, 1]
    else:
        sample_shap = shap_values[0]
    vals = pd.Series(sample_shap, index=columns)
    top = vals.abs().sort_values(ascending=False).head(5)

    st.markdown("### 🔎 Top Factors Influencing This Prediction")
    for feat, impact in top.items():
        direction = "increase" if vals[feat] > 0 else "decrease"
        st.write(f"• **{feat}** ({direction}s risk): contribution {impact:.3f}")
    fig, ax = plt.subplots()
    top.plot(kind="barh", ax=ax, color="tomato")
    ax.set_xlabel("Absolute SHAP value")
    ax.set_title("Top Feature Contributions")
    st.pyplot(fig)

    # ---------- Personalized Tips ----------
    st.markdown("### 💡 Personalized Wellness Suggestions")
    tips = []
    if cholesterol > 240:
        tips.append("Reduce saturated fats and added sugars to help lower cholesterol.")
    if resting_bp > 130:
        tips.append("Lower salt intake and keep regular physical activity to help manage blood pressure.")
    if max_hr < 100 and age < 60:
        tips.append("Moderate aerobic exercise can help improve cardiovascular fitness.")
    if oldpeak > 2:
        tips.append("Discuss your stress-test or ECG results with a healthcare provider.")
    if sex == "Male":
        tips.append("Men have slightly higher heart-disease risk; regular check-ups are important.")
    for f in top[top > 0].index:
        if "cholestoral" in f and "cholestoral" not in [t.lower() for t in tips]:
            tips.append("High cholesterol strongly influenced the prediction—consider a heart-healthy diet.")
        if "resting_blood_pressure" in f and "blood pressure" not in " ".join(tips).lower():
            tips.append("Blood pressure was a key factor—monitor and maintain it within a healthy range.")

    if tips:
        st.success("Here are tips tailored to your inputs and model results:")
        for t in tips:
            st.markdown(f"- **{t}**")
    else:
        st.success("Great job! No extra suggestions beyond maintaining a balanced lifestyle.")

    # ---------- Save to History ----------
    rec = {"date": datetime.date.today(), "probability": prob, **record}
    pd.DataFrame([rec]).to_csv(
        history_file,
        mode="a",
        header=not os.path.exists(history_file),
        index=False
    )
    st.success("Your result has been saved. You can track or manage your history below.")

    # ---------- Fixed What-if Analysis ----------
    st.markdown("### 🧮 What-if Analysis")
    new_chol = st.slider("Adjust Cholesterol (mg/dl)", 100, 600, int(cholesterol))
    new_bp   = st.slider("Adjust Resting BP (mm Hg)", 80, 200, int(resting_bp))

    # Copy original inputs but adjust cholesterol and BP
    what_if = record.copy()
    what_if["cholestoral"] = new_chol
    what_if["resting_blood_pressure"] = new_bp

    # Encode & align with training columns and scale
    df_wi = pd.DataFrame([what_if])
    df_wi_enc = pd.get_dummies(df_wi).reindex(columns=columns, fill_value=0)
    X_wi_scaled = scaler.transform(df_wi_enc)

    prob_wi = rf.predict_proba(X_wi_scaled)[0, 1]
    st.info(f"Predicted risk with these changes: **{prob_wi*100:.1f}%**")

    # ---------- Population Comparison ----------
    st.markdown("### 📊 Comparison to Dataset Averages")
    st.write(f"Average Resting BP: {train_df['resting_blood_pressure'].mean():.1f} mmHg "
             f"(yours: {resting_bp})")
    st.write(f"Average Cholesterol: {train_df['cholestoral'].mean():.1f} mg/dl "
             f"(yours: {cholesterol})")

# ------------------------------------------------------
# Cleaner Past Results with checkboxes to delete
# ------------------------------------------------------
st.markdown("## 🗂 Your Past Results")
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)

    if history_df.empty:
        st.info("No past records yet.")
    else:
        st.write("Select rows to delete and click **Delete Selected**:")

        # Add a checkbox column for deletion
        editable_df = history_df.copy()
        editable_df["Delete?"] = False

        edited = st.data_editor(
            editable_df,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True
        )

        to_delete = edited.index[edited["Delete?"]].tolist()

        if to_delete and st.button("Delete Selected"):
            remaining = history_df.drop(index=to_delete)
            remaining.to_csv(history_file, index=False)
            st.success(f"Deleted {len(to_delete)} record(s).")
            st.rerun()      # current Streamlit refresh call
else:
    st.info("No past records found.")
