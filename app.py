import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load Trained Model & Scaler
# ==========================

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==========================
# Prediction Function
# ==========================

def predict_heart_disease(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
):

    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)[0]

    probability = model.predict_proba(patient_scaled)[0]

    confidence = probability[prediction] * 100

    if prediction == 1:

        result = """
# 🔴 High Risk of Heart Disease

The model predicts that the patient may have heart disease.

Please consult a qualified healthcare professional.
"""

        tips = """
## ❤️ Heart Care Recommendations

• Exercise for at least **30 minutes daily**

• Reduce salt intake

• Eat more fruits and vegetables

• Avoid smoking

• Maintain healthy cholesterol

• Schedule regular medical checkups
"""

    else:

        result = """
# 🟢 Low Risk of Heart Disease

The patient currently appears to have a low risk of heart disease.

Continue maintaining a healthy lifestyle.
"""

        tips = """
## 💚 Healthy Lifestyle Tips

• Continue regular exercise

• Eat a balanced diet

• Get sufficient sleep

• Stay hydrated

• Continue routine health checkups
"""

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        probability,
        labels=["No Disease", "Disease"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.02, 0.08)
    )

    ax.set_title("Prediction Probability")

    return (
        result,
        f"{confidence:.2f}%",
        fig,
        tips
    )
# ==========================
# Gradio Interface
# ==========================

with gr.Blocks(
    title="Heart Disease Prediction"
) as demo:

    gr.Markdown("""
# ❤️ Heart Disease Prediction System

## Machine Learning Based Early Risk Assessment

Predict the likelihood of heart disease using patient medical information.
""")

    with gr.Accordion("ℹ️ About this Project", open=False):

        gr.Markdown("""
This application predicts whether a patient is at risk of heart disease using a **Logistic Regression Machine Learning model** trained on the **UCI Heart Disease Dataset**.

### Features

- ❤️ Heart Disease Prediction
- 📊 Prediction Probability Chart
- 📈 Confidence Score
- 💡 Personalized Prevention Tips

**Disclaimer:** This application is intended for educational purposes only.
""")

    gr.Markdown("## 📝 Patient Information")

    with gr.Row():

        with gr.Column():

            age = gr.Number(
                label="Age",
                value=50
            )

            sex = gr.Dropdown(
                choices=[("Female",0),("Male",1)],
                label="Sex",
                value=1
            )

            cp = gr.Dropdown(
                choices=[0,1,2,3],
                label="Chest Pain Type",
                value=0
            )

            trestbps = gr.Number(
                label="Resting Blood Pressure",
                value=120
            )

            chol = gr.Number(
                label="Cholesterol",
                value=200
            )

            fbs = gr.Dropdown(
                choices=[("False",0),("True",1)],
                label="Fasting Blood Sugar > 120 mg/dl",
                value=0
            )

            restecg = gr.Dropdown(
                choices=[0,1,2],
                label="Resting ECG",
                value=1
            )

        with gr.Column():

            thalach = gr.Number(
                label="Maximum Heart Rate",
                value=150
            )

            exang = gr.Dropdown(
                choices=[("No",0),("Yes",1)],
                label="Exercise Induced Angina",
                value=0
            )

            oldpeak = gr.Number(
                label="Old Peak",
                value=1.0
            )

            slope = gr.Dropdown(
                choices=[0,1,2],
                label="Slope",
                value=2
            )

            ca = gr.Dropdown(
                choices=[0,1,2,3,4],
                label="Major Vessels",
                value=0
            )

            thal = gr.Dropdown(
                choices=[0,1,2,3],
                label="Thalassemia",
                value=2
            )
    gr.Markdown("---")

    with gr.Row():

        predict_btn = gr.Button(
            "❤️ Predict Heart Disease",
            variant="primary"
        )

        clear_btn = gr.ClearButton(
            components=[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ],
            value="🗑️ Clear"
        )

    gr.Markdown("## 📋 Prediction Result")

    with gr.Row():

        with gr.Column(scale=1):

            result = gr.Markdown()

            confidence = gr.Textbox(
                label="Prediction Confidence",
                interactive=False
            )

            tips = gr.Markdown()

        with gr.Column(scale=1):

            probability_chart = gr.Plot(
                label="Prediction Probability"
            )

    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ],
        outputs=[
            result,
            confidence,
            probability_chart,
            tips
        ]
    )

    gr.Markdown("""
---

## ⚠️ Medical Disclaimer

This prediction is generated using a Machine Learning model trained on the **UCI Heart Disease Dataset**.

It is intended only for educational purposes and should **not** be considered a substitute for professional medical advice.

Always consult a qualified healthcare professional before making any medical decisions.

Made with ❤️ using **Python**, **Scikit-learn**, **Matplotlib**, and **Gradio**.
""")
    
    demo.launch()