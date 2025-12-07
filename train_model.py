import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create model folder if not exists
os.makedirs('model', exist_ok=True)
# Load dataset
data = pd.read_csv('data/Training.csv')

# Drop duplicate rows (if any)
data = data.drop_duplicates()

# Handle missing values — replace NaN with 0
data = data.fillna(0)

# Separate features (X) and target (y)
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, 'model/model.pkl')
print("\n💾 Model saved successfully as model/model.pkl")
