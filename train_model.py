import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load data
df = pd.read_csv('creditcard.csv')

# 2. Preprocessing
X = df.drop(['Class'], axis=1)
y = df['Class']

# Feature scaling (important for ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluation
preds = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# 6. Save model and scaler
joblib.dump(model, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('Model and scaler saved!') 