import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
model_weights = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate weight φ = (Accuracy * F1 score) / Precision
    weight = (accuracy * f1) / precision if precision > 0 else 0
    model_weights[name] = weight
    
    results.append([name, accuracy, precision, recall, f1, weight])
    
    # Save individual model
    with open(f'model_{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: model_{name.replace(' ', '_').lower()}.pkl")

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score", "Weight"])

# Save scaler and metadata
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model_weights, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

with open('continuous_cols.pkl', 'wb') as f:
    pickle.dump(continuous_cols, f)

# Calculate total number of models
n_models = len(models)
with open('n_models.pkl', 'wb') as f:
    pickle.dump(n_models, f)

print("\nAll models and components saved successfully!")
print(f"\nTotal models: {n_models}")
print("\nModel Performance and Weights:")
print(results_df)
print("\nModel Weights (φ):")
for name, weight in model_weights.items():
    print(f"{name}: {weight:.4f}")