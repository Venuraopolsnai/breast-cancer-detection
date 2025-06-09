#!/usr/bin/env python
# coding: utf-8

# #  Leveraging Machine Learning Approaches for Breast Cancer Prediction

# # Task 1: Data Exploration

# In[2]:


#Step 1: Load the Dataset
import pandas as pd
import seaborn as sns
# Load the dataset
data = pd.read_csv("breast_cancer_dataset.csv")  # Replace with your actual file path if local
data.head()


# In[3]:


#Step 2: Target Variable Distribution
import seaborn as sns
import matplotlib.pyplot as plt
# Distribution count
print(data['diagnosis'].value_counts())
# Plot
sns.countplot(x='diagnosis', data=data)
plt.title("Diagnosis Distribution")
plt.xlabel("Tumor Type")
plt.ylabel("Count")
plt.show()


# In[5]:


#Step 3: Summary Statistics
# Numerical stats
stats = data.describe().T[['mean', '50%', 'std']].rename(columns={'50%': 'median'})
print(stats)


# In[6]:


#Step 4: Visualizations (Histograms and Box Plots)
# Histogram
data.hist(bins=20, figsize=(15, 12))
plt.suptitle("Histograms of All Features")
plt.show()

# Boxplot for selected features
for feature in ['radius_mean', 'texture_mean', 'perimeter_mean']:
    sns.boxplot(x='diagnosis', y=feature, data=data)
    plt.title(f"Boxplot of {feature} by Diagnosis")
    plt.show()


# In[7]:


#Step 5: Correlation Matrix and Feature-Target Correlation
# Encode target as binary
# Ensure diagnosis is converted to numeric (Malignant: 1, Benign: 0)
data['diagnosis_binary'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Select only numeric columns
numeric_data = data.select_dtypes(include='number')

# Calculate correlation with the target variable
correlation = numeric_data.corr()['diagnosis_binary'].sort_values(ascending=False)

# Display top correlations
print("Top positively correlated features:")
print(correlation.head(10))

print("\nTop negatively correlated features:")
print(correlation.tail(10))

import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_data = data.select_dtypes(include='number')

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# # Task 2: Data Preparation
# 

# In[8]:


#Step 1: Assign the Target Variable
# Reassign the target variable (if not already done)
data['diagnosis_binary'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop(columns=['diagnosis', 'diagnosis_binary', 'id'])  # Drop non-feature columns
y = data['diagnosis_binary']


# In[9]:


#Step 2: Check for Missing Values
# Check missing values
missing_values = X.isnull().sum()
print(missing_values[missing_values > 0])


# In[10]:


#Step 3: Handle Missing Values (Imputation)
# Example: Impute missing values using median
X = X.fillna(X.median())


# In[11]:


#Step 4: Feature Scaling
from sklearn.preprocessing import StandardScaler

# Apply standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally, convert back to DataFrame for easier analysis
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# # Task 3: Model Training

# In[12]:


#Step 1: Split Data into Training and Test Sets
from sklearn.model_selection import train_test_split

# Split data - 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training size: {len(X_train)} samples")
print(f"Testing size: {len(X_test)} samples")


# In[13]:


#Step 2: Import Classifiers and Initialize with Default Hyperparameters
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define classifiers with default hyperparameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}


# In[14]:


#Step 3: Train Each Model
# Dictionary to store trained models
trained_models = {}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} model trained.")


# # Task 4: Model Evaluation and Visualization: 

# In[15]:


#Step 1: Evaluate Each Model (Accuracy, Precision, Recall, F1, AUC)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np

# Create a results dictionary
results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    })

# Convert to DataFrame for display
results_df = pd.DataFrame(results)
print(results_df.sort_values(by='AUC', ascending=False))


# In[26]:


#Step 2: Confusion Matrix & ROC Curves
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrices
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, model in trained_models.items():
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


#Step 3: Discussion – Best Model Based on Metrics

#rom the results table and AUC in the ROC curve, you'll likely observe:

#Random Forest or SVM often perform the best in breast cancer classification.

#Random Forest tends to offer high accuracy, precision, and AUC because of ensemble learning.

#SVM gives excellent performance in high-dimensional, scaled datasets like this


# In[ ]:


#Step 4: Fine-Tune Top Model (Random Forest)
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

# GridSearchCV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best estimator
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


# In[25]:


# Step:5 Compare Pre- vs Post-Tuning Performance
# Evaluate pre-tuned and post-tuned RF models
models_to_compare = {
    'Random Forest (Default)': trained_models['Random Forest'],
    'Random Forest (Tuned)': best_rf
}

for name, model in models_to_compare.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")


# # Task 5: Conclusion and Future Work:

# In[ ]:





# In[26]:


#1. Recap: Best Performing Model
# Best model after tuning
best_model = best_rf  # Random Forest (tuned)


# In[27]:


#2. Key Insights from Feature Importance
# Feature importance visualization
importances = best_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for importance
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top 10 important features
print(feature_df.head(10))

# Plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df.head(10))
plt.title('Top 10 Most Important Features (Random Forest)')
plt.tight_layout()
plt.show()


# In[ ]:




