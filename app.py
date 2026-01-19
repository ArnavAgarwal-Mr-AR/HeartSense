import numpy as np
import pandas as pd
import gradio as gr
import pickle

# Load all saved components
with open('models.pkl', 'rb') as f:
    trained_models = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ratios.pkl', 'rb') as f:
    ratios = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('continuous_cols.pkl', 'rb') as f:
    continuous_cols = pickle.load(f)

def predict_heart_disease(age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope):
    # Encode categorical inputs
    sex = 1 if sex == "Male" else 0  
    chest_pain_type_mapping = {"ASY": 1, "ATA": 2, "NAP": 3, "TA": 4}
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]  
    resting_ecg_mapping = {"Normal": 1, "ST": 2, "LVH": 3}
    resting_ecg = resting_ecg_mapping[resting_ecg]  
    exercise_angina_mapping = {"No": 0, "Yes": 1}
    exercise_angina = exercise_angina_mapping[exercise_angina]  
    st_slope_mapping = {"Up": 1, "Flat": 2, "Down": 3}
    st_slope = st_slope_mapping[st_slope]  
    
    # Create input array
    input_data = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope, fasting_bs]])
    input_df = pd.DataFrame(input_data, columns=["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"])
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale continuous features
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    # Get predictions from all models
    predictions = {name: model.predict(input_df)[0] for name, model in trained_models.items()}
    
    # Calculate weighted sum
    total_weight = sum(prediction * ratios[name] for name, prediction in predictions.items())
    
    # Determine output based on total weight
    if total_weight > 3.5:
        output = "⚠️ Warning: You might be at risk of heart disease. Please consult a healthcare professional for proper evaluation."
    else:
        output = "✓ Based on the input parameters, your risk appears to be lower. However, always consult with a healthcare professional for accurate diagnosis."
    
    # Add individual model predictions for transparency
    model_results = "\n\nIndividual Model Predictions:\n"
    for name, pred in predictions.items():
        result = "Disease" if pred == 1 else "No Disease"
        model_results += f"- {name}: {result}\n"
    
    return output + model_results

# Create Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50),
        gr.Number(label="Resting BP (mm Hg)", value=120),
        gr.Number(label="Cholesterol (mg/dL)", value=200),
        gr.Number(label="Max Heart Rate", value=150),
        gr.Number(label="Oldpeak (0-2)", value=0),
        gr.Number(label="Fasting Blood Sugar (0 or 1)", value=0),
        gr.Radio(["Female", "Male"], label="Sex", value="Male"),
        gr.Radio(["ASY", "ATA", "NAP", "TA"], label="Chest Pain Type", value="ASY"),
        gr.Radio(["Normal", "ST", "LVH"], label="Resting ECG", value="Normal"),
        gr.Radio(["No", "Yes"], label="Exercise Angina", value="No"),
        gr.Radio(["Up", "Flat", "Down"], label="ST Slope", value="Flat"),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=10),
    title="❤️ Heart Disease Prediction System",
    description="Enter patient details to predict heart disease risk using an ensemble of 7 machine learning models. This tool is for educational purposes only and should not replace professional medical advice.",
    examples=[
        [55, 130, 250, 140, 1.0, 0, "Male", "ASY", "Normal", "No", "Flat"],
        [45, 110, 180, 160, 0, 0, "Female", "NAP", "Normal", "No", "Up"],
    ]
)

if __name__ == "__main__":
    iface.launch()