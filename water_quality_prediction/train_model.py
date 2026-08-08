import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv('water_potability.csv')
print("Dataset loaded successfully.")

# 2. Fill Missing Values (Imputation using Median)
df['ph'].fillna(df['ph'].median(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].median(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].median(), inplace=True)

print("Missing values handled.")

# 3. Separate Features and Target
X = df.drop('Potability', axis=1)
y = df['Potability']

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test_scaled)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save Model and Scaler for App Usage
joblib.dump(model, 'water_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved as 'water_model.pkl' and scaler saved as 'scaler.pkl'.")