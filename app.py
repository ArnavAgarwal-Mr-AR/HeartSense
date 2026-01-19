import numpy as np
import pandas as pd
import gradio as gr
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load model weights and metadata
with open('model_weights.pkl', 'rb') as f:
    model_weights = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('continuous_cols.pkl', 'rb') as f:
    continuous_cols = pickle.load(f)

with open('n_models.pkl', 'rb') as f:
    n_models = pickle.load(f)

# Load individual models
models = {}
model_names = [
    "logistic_regression",
    "decision_tree", 
    "gradient_boosting",
    "knn",
    "svm",
    "random_forest",
    "xgboost"
]

model_display_names = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "gradient_boosting": "Gradient Boosting",
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

for model_name in model_names:
    with open(f'model_{model_name}.pkl', 'rb') as f:
        models[model_display_names[model_name]] = pickle.load(f)

print(f"Loaded {len(models)} models successfully!")

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    # Determine color based on risk level
    if probability < 30:
        color = "#10b981"  # Green
        risk_level = "Low Risk"
    elif probability < 50:
        color = "#f59e0b"  # Orange
        risk_level = "Moderate Risk"
    elif probability < 70:
        color = "#ef4444"  # Red
        risk_level = "High Risk"
    else:
        color = "#dc2626"  # Dark Red
        risk_level = "Very High Risk"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>{risk_level}</b>", 'font': {'size': 24, 'color': color}},
        number = {'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 50], 'color': '#fef3c7'},
                {'range': [50, 70], 'color': '#fee2e2'},
                {'range': [70, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Arial"}
    )
    
    return fig

def create_summary_text(probability, weighted_sum, threshold, prediction_result, model_predictions, model_weights):
    """Create formatted HTML summary"""
    if weighted_sum > threshold:
        status_color = "#dc2626"
        status_icon = "⚠️"
        recommendation = """
        <div style='background-color: #fee2e2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;'>
            <h3 style='color: #991b1b; margin-top: 0;'>⚠️ Action Recommended</h3>
            <ul style='color: #7f1d1d; margin-bottom: 0;'>
                <li style='color: #000000;'>Consult a healthcare professional for thorough evaluation</li>
                <li style='color: #000000;'>Consider scheduling a cardiac check-up</li>
                <li style='color: #000000;'>Monitor your symptoms closely</li>
                <li style='color: #000000;'>Maintain a heart-healthy lifestyle</li>
            </ul>
        </div>
        """
    else:
        status_color = "#10b981"
        status_icon = "✓"
        recommendation = """
        <div style='background-color: #d1fae5; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;'>
            <h3 style='color: #065f46; margin-top: 0;'>✓ Lower Risk Detected</h3>
            <ul style='color: #064e3b; margin-bottom: 0;'>
                <li style='color: #000000;'>Continue maintaining a healthy lifestyle</li>
                <li style='color: #000000;'>Regular exercise and balanced diet recommended</li>
                <li style='color: #000000;'>Periodic health check-ups are still important</li>
                <li style='color: #000000;'>Stay aware of any changes in symptoms</li>
            </ul>
        </div>
        """
    
    # Create model predictions breakdown
    predictions_breakdown = ""
    disease_count = sum(1 for p in model_predictions.values() if p == 1)
    safe_count = len(model_predictions) - disease_count
    
    html = f"""
    <div style='font-family: Arial, sans-serif; padding: 20px;'>      
        {recommendation}
        
        <div style='background-color: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 20px;'>
            <h4 style='color: #374151; margin-top: 0;'>How This Works</h4>
            <p style='color: #6b7280; font-size: 14px; line-height: 1.6; margin-bottom: 0;'>
                This prediction uses an ensemble of 7 different machine learning models. Each model's prediction is weighted based 
                on its performance metrics. The final decision is made when the weighted sum exceeds the threshold.
            </p>
        </div>
        
        <div style='background-color: #fffbeb; padding: 12px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #f59e0b;'>
            <p style='color: #92400e; font-size: 13px; margin: 0;'>
                Medical Disclaimer: This tool is for educational purposes only and should not replace 
                professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers 
                for medical decisions.
            </p>
        </div>
    </div>
    """
    
    return html

def predict_heart_disease(age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope):
    # Encode categorical inputs using LabelEncoder compatible mappings
    # These must match what LabelEncoder produced during training
    sex_mapping = {"Male": 1, "Female": 0}
    sex = sex_mapping.get(sex, 0)
    
    # Convert fasting_bs from string to integer
    fasting_bs = int(fasting_bs)
    
    # For string categorical values, LabelEncoder sorts them alphabetically:
    # ChestPainType: ["ASY", "ATA", "NAP", "TA"] -> 0, 1, 2, 3
    chest_pain_type_mapping = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
    chest_pain_type = chest_pain_type_mapping.get(chest_pain_type, 0)
    
    # RestingECG: ["LVH", "Normal", "ST"] -> 0, 1, 2
    resting_ecg_mapping = {"LVH": 0, "Normal": 1, "ST": 2}
    resting_ecg = resting_ecg_mapping.get(resting_ecg, 1)
    
    # ExerciseAngina: ["N", "Y"] -> 0, 1
    exercise_angina_mapping = {"No": 0, "Yes": 1}
    exercise_angina = exercise_angina_mapping.get(exercise_angina, 0)
    
    # ST_Slope: ["Down", "Flat", "Up"] -> 0, 1, 2
    st_slope_mapping = {"Down": 0, "Flat": 1, "Up": 2}
    st_slope = st_slope_mapping.get(st_slope, 2)
    
    # Create input dataframe matching exact training column order
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_type],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })
    
    # Ensure columns match feature_names order exactly
    input_df = input_df[feature_names]
    
    # Scale continuous features
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    # Get predictions from all models
    model_predictions = {}
    weighted_sum = 0
    total_weights = sum(model_weights.values())
    
    for name, model in models.items():
        prediction = model.predict(input_df)[0]
        weight = model_weights[name]
        model_predictions[name] = prediction
        weighted_sum += prediction * weight
    
    # Calculate probability and threshold
    threshold = n_models / 2
    # Probability is the weighted sum normalized by total possible weighted sum
    max_possible_sum = sum(model_weights.values())
    probability = (weighted_sum / max_possible_sum) * 100
    
    # Determine final prediction
    if weighted_sum > threshold:
        result = "High Risk Detected"
    else:
        result = "Low Risk - You Are Safe"
    
    # Create visualizations
    gauge_chart = create_gauge_chart(probability)
    summary_html = create_summary_text(probability, weighted_sum, threshold, result, model_predictions, model_weights)
    return summary_html, gauge_chart

# Create Gradio interface with custom CSS
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.output-html {
    border: none !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # HeartSense - Heart Disease Prediction System
        ### Advanced ML Ensemble for Cardiovascular Risk Assessment
        Enter patient details below to get a comprehensive risk analysis using 7 different machine learning models.
        """
    )
    
    with gr.Column():
        with gr.Row(scale=1):
            with gr.Column(scale=1):
                gr.Markdown("### Patient Information")
                age = gr.Number(label="Age", value=50)
                sex = gr.Radio(["Female", "Male"], label="Sex", value="Male")
                fasting_bs = gr.Radio(["0", "1"], label="Fasting Blood Sugar", value="0")
            
            with gr.Column(scale=1):
                gr.Markdown("### Cardiac Measurements")
                resting_bp = gr.Number(label="Resting BP (mm Hg)", value=120)
                cholesterol = gr.Number(label="Cholesterol (mg/dL)", value=200)
                max_hr = gr.Number(label="Max Heart Rate", value=150)
                oldpeak = gr.Number(label="Oldpeak", value=0)
            
            with gr.Column(scale=1):
                gr.Markdown("### Clinical Indicators")
                chest_pain_type = gr.Radio(["ASY", "ATA", "NAP", "TA"], label="Chest Pain Type", value="ASY")
                resting_ecg = gr.Radio(["Normal", "ST", "LVH"], label="Resting ECG", value="Normal")
                exercise_angina = gr.Radio(["No", "Yes"], label="Exercise Angina", value="No")
                st_slope = gr.Radio(["Up", "Flat", "Down"], label="ST Slope", value="Flat")
            
        predict_btn = gr.Button("Analyze Risk", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gauge_output = gr.Plot(label="Risk Gauge")
            summary_output = gr.HTML(label="Summary")
    
    gr.Markdown(
        """
        ### Example Cases
        Try these example inputs to see how the system works:
        """
    )
    
    gr.Examples(
        examples=[
            [55, 130, 250, 140, 1.0, "0", "Male", "ASY", "Normal", "No", "Flat"],
            [45, 110, 180, 160, 0, "0", "Female", "NAP", "Normal", "No", "Up"],
            [60, 140, 280, 130, 2.0, "1", "Male", "ASY", "ST", "Yes", "Down"],
        ],
        inputs=[age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope],
    )
    
    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope],
        outputs=[summary_output, gauge_output]
    )

if __name__ == "__main__":
    iface.launch()