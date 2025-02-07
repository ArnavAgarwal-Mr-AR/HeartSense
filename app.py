import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
mydata = pd.read_csv("dataset.csv")


# Remove duplicates
mydata = mydata.drop_duplicates()


# Outlier removal using IQR
variables = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
for col in variables:
    q1 = mydata[col].quantile(0.25)
    q3 = mydata[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mydata = mydata[(mydata[col] >= lower_bound) & (mydata[col] <= upper_bound)]


# Label Encoding
label_encoders = {}
for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
    le = LabelEncoder()
    mydata[col] = le.fit_transform(mydata[col])
    label_encoders[col] = le


# Splitting Data
X = mydata.drop("HeartDisease", axis=1)
y = mydata["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# Scaling continuous variables
scaler = StandardScaler()
continuous_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])


# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, accuracy, precision, recall, f1])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score"])

# Calculate metrics
metrics = results_df[["Model", "Accuracy", "F1_Score", "Precision"]].copy()
metrics["Metric"] = (metrics["Accuracy"] * metrics["F1_Score"]) / metrics["Precision"]

# Calculate average metric and ratios
avg_metric = metrics["Metric"].mean()
metrics["Ratio"] = metrics["Metric"] / avg_metric

# Create a dictionary for ratios
ratios = dict(zip(metrics["Model"], metrics["Ratio"]))



trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

def predict_heart_disease(age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope):
    sex = 1 if sex == "Male" else 0  
    chest_pain_type_mapping = {"ASY": 1, "ATA": 2, "NAP": 3, "TA": 4}
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]  
    resting_ecg_mapping = {"Normal": 1, "ST": 2, "LVH": 3}
    resting_ecg = resting_ecg_mapping[resting_ecg]  
    exercise_angina_mapping = {"No": 0, "Yes": 1}
    exercise_angina = exercise_angina_mapping[exercise_angina]  
    st_slope_mapping = {"Up": 1, "Flat": 2, "Down": 3}
    st_slope = st_slope_mapping[st_slope]  
    input_data = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak, sex, chest_pain_type, resting_ecg, exercise_angina, st_slope, fasting_bs]])
    input_df = pd.DataFrame(input_data, columns=["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"])
    input_df = input_df[X_train.columns] 
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    predictions = {name: model.predict(input_df)[0] for name, model in trained_models.items()}
    # Calculate weight values and sum them
    total_weight = sum(prediction * ratios[name] for name, prediction in predictions.items())
    # Determine output based on total weight
    if total_weight > 3.5:
        output = "You might get a heart stroke take precautions"
    else:
        output = "You are safe"
    return output


iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="RestingBP"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="MaxHR"),
        gr.Number(label="Oldpeak (0-2)"),
        gr.Number(label="FastingBS"),
        gr.Radio(["Female", "Male"], label="Sex"),
        gr.Radio(["ASY", "ATA", "NAP", "TA"], label="Chest Pain Type"),
        gr.Radio(["Normal", "ST", "LVH"], label="Resting ECG"),
        gr.Radio(["No", "Yes"], label="Exercise Angina"),
        gr.Radio(["Up", "Flat", "Down"], label="ST Slope"),
    ],
    outputs=gr.Text(),
    title="Heart Disease Prediction",
    description="Enter patient details to predict heart disease risk using multiple models."
)
iface.launch()