import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Loan_approval_data_2025.csv")

# Example target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_xgboost_pipeline.pkl")

print("Model retrained and saved successfully!")