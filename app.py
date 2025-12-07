from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)
app.secret_key = "dev-secret-change-in-production"

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"
TRAINING_CSV = ROOT / "data" / "Training.csv"

# Auto CSS versioning
def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, 'static', filename)
            if os.path.exists(file_path):
                values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

app.jinja_env.globals['dated_url_for'] = dated_url_for

# Basic checks
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model.py first.")
if not TRAINING_CSV.exists():
    raise FileNotFoundError(f"Training CSV not found at {TRAINING_CSV}. Make sure it exists.")

# Load model
model = joblib.load(MODEL_PATH)

# Load training feature columns
df_train = pd.read_csv(TRAINING_CSV)
feature_columns = df_train.drop(columns=['prognosis']).columns.tolist()

# List of symptom-like columns
symptom_candidates = [c for c in feature_columns if c not in ('age', 'gender', 'weight', 'height', 'blood_pressure', 'body_temperature', 'blood_sugar')]

# Dummy doctors
DOCTORS = [
    {"name": "Dr. Priya Sharma", "specialty": "General Physician", "phone": "+91-9876543210", "distance": "1.2 km"},
    {"name": "Dr. Rohit Patel", "specialty": "Pulmonologist", "phone": "+91-9123456780", "distance": "2.5 km"},
    {"name": "Dr. Anjali Mehta", "specialty": "Neurologist", "phone": "+91-9988776655", "distance": "3.0 km"},
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html', vocab=symptom_candidates[:200])

def build_input_row(form):
    row = pd.Series({c: 0 for c in feature_columns}, dtype=float)
    numeric_fields = ['age', 'height', 'weight', 'blood_pressure', 'body_temperature', 'blood_sugar']
    for nf in numeric_fields:
        val = form.get(nf)
        if val:
            try:
                row[nf] = float(val)
            except:
                pass
    if 'gender' in feature_columns:
        g = form.get('gender','').strip().lower()
        mapping = {'male':1,'m':1,'female':0,'f':0,'other':0.5}
        row['gender'] = mapping.get(g,0)

    raw_symptoms = form.get('symptoms_text','').strip() or form.get('symptoms_list','').strip()
    symptom_list = [s.strip().lower() for s in raw_symptoms.replace(',', ';').split(';') if s.strip()]
    for s in symptom_list:
        if s in feature_columns:
            row[s] = 1
        else:
            norm = s.replace(' ','_').replace('-','_')
            if norm in feature_columns:
                row[norm] = 1
            else:
                for col in feature_columns:
                    if s in col.lower():
                        row[col] = 1
    input_df = pd.DataFrame([row], columns=feature_columns).fillna(0)
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name','').strip()
        age = request.form.get('age','').strip()
        if not name or not age:
            flash("Please provide at least Name and Age before predicting.")
            return redirect(url_for('index'))

        input_df = build_input_row(request.form).astype(float)

        if hasattr(model,"predict_proba"):
            probs = model.predict_proba(input_df)[0]
            classes = model.classes_
            idx = np.argsort(probs)[::-1][:3]
            top = [{"disease": classes[i], "confidence": float(probs[i])} for i in idx]
        else:
            pred = model.predict(input_df)[0]
            top = [{"disease": pred, "confidence":1.0}]

        df_meta = pd.read_csv(TRAINING_CSV)
        disease_info = {}
        for p in top:
            d = p['disease']
            matches = df_meta[df_meta['prognosis']==d]
            if not matches.empty:
                disease_info[d] = {
                    "description": f"Predicted disease: {d}. This is a common diagnosis in the dataset.",
                    "causes": "",
                    "precautions": [],
                    "common_meds": [],
                    "life_threatening": 0
                }
            else:
                disease_info[d] = {"description":"No description available.","causes":"","precautions":[],"common_meds":[],"life_threatening":0}

        user = {
            "name": name,
            "age": age,
            "gender": request.form.get('gender',''),
            "height": request.form.get('height',''),
            "weight": request.form.get('weight',''),
            "allergies": request.form.get('allergies',''),
            "surgeries": request.form.get('surgeries',''),
            "family_history": request.form.get('family_history',''),
        }

        return render_template('result.html', user=user, predictions=top, disease_info=disease_info, doctors=DOCTORS)
    except Exception as e:
        print("Error in /predict:", e)
        flash("An error occurred while predicting. Check server logs.")
        return redirect(url_for('index'))

@app.route('/doctors')
def doctors():
    return render_template('doctors.html', doctors=DOCTORS)

@app.route('/api/symptoms')
def api_symptoms():
    q = (request.args.get('q') or "").lower().strip()
    suggestions = [s for s in symptom_candidates if q in s.lower()][:30]
    return {"suggestions": suggestions}

if __name__=="__main__":
    app.run(debug=True)
