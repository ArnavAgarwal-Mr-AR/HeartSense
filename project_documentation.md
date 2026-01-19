# Heart Disease Prediction System - Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Detailed File Descriptions](#detailed-file-descriptions)
4. [Setup Instructions](#setup-instructions)
5. [Technical Details](#technical-details)

---

## 🎯 Project Overview

This is a machine learning-based heart disease prediction system that uses an ensemble of 7 different ML models to assess cardiovascular risk. The system is deployed on Hugging Face Spaces using Gradio for the user interface.

**Models Used:**
- Logistic Regression
- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## 📁 File Structure

```
heart-disease-prediction/
├── train_models.py              # Training script (run locally)
├── app.py                       # Gradio web application (deploy to HF)
├── requirements.txt             # Python dependencies
├── README.md                    # Project description (optional)
├── eda_heart.csv               # Training dataset (local only)
│
├── model_logistic_regression.pkl    # Trained Logistic Regression model
├── model_decision_tree.pkl          # Trained Decision Tree model
├── model_gradient_boosting.pkl      # Trained Gradient Boosting model
├── model_knn.pkl                    # Trained KNN model
├── model_svm.pkl                    # Trained SVM model
├── model_random_forest.pkl          # Trained Random Forest model
├── model_xgboost.pkl                # Trained XGBoost model
│
├── scaler.pkl                   # StandardScaler for feature normalization
├── model_weights.pkl            # Calculated weights for each model
├── feature_names.pkl            # Column names/order from training
├── continuous_cols.pkl          # List of continuous feature columns
└── n_models.pkl                # Total number of models (7)
```

---

## 📄 Detailed File Descriptions

### **Python Scripts**

#### `train_models.py`
**Purpose:** Training script to be run locally before deployment

**What it does:**
- Loads the `eda_heart.csv` dataset
- Performs data preprocessing:
  - Removes duplicate entries
  - Handles outliers using IQR method
  - Encodes categorical variables
  - Splits data into train/test sets (70/30)
  - Scales continuous features using StandardScaler
- Trains all 7 machine learning models
- Calculates model weights using formula: `φ = (Accuracy × F1 Score) / Precision`
- Saves all models and preprocessing components as pickle files
- Displays performance metrics for each model

**When to run:** Once locally before deploying to Hugging Face Spaces

**Output files:** All `.pkl` files listed below

---

#### `app.py`
**Purpose:** Main Gradio application for deployment on Hugging Face Spaces

**What it does:**
- Loads all trained models and preprocessing components
- Provides interactive web interface for user input
- Accepts patient health parameters
- Performs predictions using all 7 models
- Calculates weighted ensemble prediction
- Displays results with:
  - Risk probability gauge chart
  - Model contributions bar chart
  - Detailed risk assessment summary
  - Personalized recommendations

**When to use:** Deploy this to Hugging Face Spaces

**Dependencies:** See `requirements.txt`

---

#### `requirements.txt`
**Purpose:** Specifies all Python package dependencies

**Contents:**
```
pandas==2.0.3           # Data manipulation
numpy==1.24.3           # Numerical computations
scikit-learn==1.3.0     # ML models and preprocessing
xgboost==2.0.3          # XGBoost classifier
gradio==4.44.0          # Web interface
huggingface_hub==0.23.0 # HuggingFace integration
plotly==5.18.0          # Interactive visualizations
```

**Note:** Version pinning ensures compatibility and reproducibility

---

### **Pickle Files (.pkl)**

#### **Model Files**

##### `model_logistic_regression.pkl`
- **Type:** Trained scikit-learn LogisticRegression model
- **Size:** ~5-10 KB
- **Purpose:** Binary classification using logistic function
- **Characteristics:** Linear decision boundary, interpretable coefficients

##### `model_decision_tree.pkl`
- **Type:** Trained scikit-learn DecisionTreeClassifier model
- **Size:** ~50-100 KB
- **Purpose:** Tree-based classification with hierarchical splits
- **Characteristics:** Non-linear, prone to overfitting, interpretable

##### `model_gradient_boosting.pkl`
- **Type:** Trained scikit-learn GradientBoostingClassifier model
- **Size:** ~500 KB - 2 MB
- **Purpose:** Ensemble of weak decision trees built sequentially
- **Characteristics:** High accuracy, handles complex patterns

##### `model_knn.pkl`
- **Type:** Trained scikit-learn KNeighborsClassifier model (k=5)
- **Size:** ~10-50 KB
- **Purpose:** Classification based on 5 nearest neighbors
- **Characteristics:** Instance-based learning, no training phase

##### `model_svm.pkl`
- **Type:** Trained scikit-learn SVC model (linear kernel)
- **Size:** ~50-200 KB
- **Purpose:** Finds optimal hyperplane for class separation
- **Characteristics:** Effective in high dimensions, margin-based

##### `model_random_forest.pkl`
- **Type:** Trained scikit-learn RandomForestClassifier model (100 trees)
- **Size:** ~2-5 MB
- **Purpose:** Ensemble of decision trees with bagging
- **Characteristics:** Robust, reduces overfitting, feature importance

##### `model_xgboost.pkl`
- **Type:** Trained XGBoost XGBClassifier model
- **Size:** ~500 KB - 2 MB
- **Purpose:** Optimized gradient boosting with regularization
- **Characteristics:** High performance, handles missing values

---

#### **Preprocessing Files**

##### `scaler.pkl`
- **Type:** scikit-learn StandardScaler object
- **Size:** ~2-5 KB
- **Purpose:** Normalizes continuous features to zero mean and unit variance
- **Features scaled:** Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- **Formula:** `z = (x - μ) / σ`
- **Critical:** Must use same scaler for consistency between training and prediction

##### `feature_names.pkl`
- **Type:** Python list of strings
- **Size:** <1 KB
- **Purpose:** Stores exact column order from training data
- **Contents:** 
  ```python
  ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 
   'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 
   'ST_Slope', 'FastingBS']
  ```
- **Why needed:** Ensures input data columns match training data order

##### `continuous_cols.pkl`
- **Type:** Python list of strings
- **Size:** <1 KB
- **Purpose:** Identifies which columns need scaling
- **Contents:**
  ```python
  ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
  ```
- **Why needed:** Tells scaler which columns to transform

---

#### **Model Configuration Files**

##### `model_weights.pkl`
- **Type:** Python dictionary
- **Size:** <1 KB
- **Purpose:** Stores calculated weight (φ) for each model
- **Structure:**
  ```python
  {
    'Logistic Regression': 0.8234,
    'Decision Tree': 0.7891,
    'Gradient Boosting': 0.9123,
    'KNN': 0.8456,
    'SVM': 0.8678,
    'Random Forest': 0.8901,
    'XGBoost': 0.9234
  }
  ```
- **Formula:** `φ = (Accuracy × F1 Score) / Precision`
- **Usage:** Weights each model's prediction in the ensemble

##### `n_models.pkl`
- **Type:** Python integer
- **Size:** <1 KB
- **Purpose:** Stores total number of models in ensemble
- **Value:** 7
- **Usage:** Used to calculate threshold (n/2 = 3.5) for final prediction

---

### **Data Files**

#### `eda_heart.csv`
- **Type:** CSV dataset
- **Size:** ~50-100 KB
- **Purpose:** Training data for model development
- **Note:** Only needed locally for training, NOT uploaded to Hugging Face Spaces
- **Columns:**
  - Age (years)
  - Sex (M/F)
  - ChestPainType (ASY/ATA/NAP/TA)
  - RestingBP (mm Hg)
  - Cholesterol (mg/dL)
  - FastingBS (0/1)
  - RestingECG (Normal/ST/LVH)
  - MaxHR (bpm)
  - ExerciseAngina (Y/N)
  - Oldpeak (ST depression)
  - ST_Slope (Up/Flat/Down)
  - HeartDisease (0/1) - Target variable

---

## 🚀 Setup Instructions

### **Step 1: Local Training**

1. Ensure you have the dataset `eda_heart.csv` in your working directory
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost
   ```
3. Run training script:
   ```bash
   python train_models.py
   ```
4. Verify all `.pkl` files are created (14 files total)

### **Step 2: Hugging Face Spaces Deployment**

1. Create a new Space on Hugging Face:
   - Select **Gradio** as SDK
   - Choose Python **3.10** or **3.11**

2. Upload these files to your Space:
   ```
   ├── app.py
   ├── requirements.txt
   ├── model_logistic_regression.pkl
   ├── model_decision_tree.pkl
   ├── model_gradient_boosting.pkl
   ├── model_knn.pkl
   ├── model_svm.pkl
   ├── model_random_forest.pkl
   ├── model_xgboost.pkl
   ├── scaler.pkl
   ├── model_weights.pkl
   ├── feature_names.pkl
   ├── continuous_cols.pkl
   └── n_models.pkl
   ```

3. **DO NOT upload** `eda_heart.csv` or `train_models.py`

4. Space will automatically build and deploy

---

## 🔬 Technical Details

### **Prediction Pipeline**

1. **Input Collection:** User enters 11 health parameters
2. **Encoding:** Categorical variables converted to numerical
3. **Scaling:** Continuous features normalized using saved scaler
4. **Model Predictions:** Each of 7 models makes binary prediction (0/1)
5. **Weighted Sum:** Calculate `Σ(prediction_i × weight_i)`
6. **Threshold Check:** If weighted_sum > 3.5, predict disease
7. **Probability:** Calculate risk percentage
8. **Visualization:** Generate gauge and contribution charts

### **Weight Calculation Formula**

For each model:
```
φ = (Accuracy × F1 Score) / Precision

where:
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
```

### **Final Prediction Logic**

```python
weighted_sum = Σ(model_prediction_i × weight_i)
threshold = n_models / 2 = 7 / 2 = 3.5

if weighted_sum > threshold:
    prediction = "High Risk Detected"
else:
    prediction = "Low Risk - You Are Safe"

probability = (weighted_sum / sum(all_weights)) × 100
```

---

## 📊 File Size Summary

| File Type | Approximate Size | Count |
|-----------|-----------------|-------|
| Model files (.pkl) | 5 MB - 10 MB total | 7 |
| Preprocessing files (.pkl) | 10 KB total | 3 |
| Configuration files (.pkl) | 2 KB total | 2 |
| Python scripts (.py) | 20 KB total | 2 |
| Requirements (.txt) | 1 KB | 1 |
| **Total for deployment** | **~5-10 MB** | **13 files** |

---

## ⚠️ Important Notes

1. **Version Compatibility:** Keep exact versions in `requirements.txt` to avoid conflicts
2. **Python Version:** Use Python 3.10 or 3.11 (not 3.12/3.13)
3. **File Order:** All pickle files must be uploaded before app starts
4. **Model Consistency:** Never retrain individual models separately - always retrain all together
5. **Scaler Critical:** Using wrong scaler will produce incorrect predictions

---

## 🔄 Updating Models

To update the models:

1. Modify `train_models.py` as needed
2. Run training script locally
3. Replace ALL `.pkl` files in Hugging Face Space
4. Never replace individual model files without replacing others

---

## 📝 License & Disclaimer

This tool is for **educational purposes only** and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.