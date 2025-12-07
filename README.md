Disease Prediction System 🩺 | Machine Learning + Flask

A machine learning–based web application that predicts the top 3 possible diseases based on user-entered symptoms. The system uses a Random Forest Classifier, a custom symptom-to-feature mapping engine, and a Flask web interface to deliver accurate, real-time health predictions.

⭐ Features

🔍 Predicts Top 3 diseases with confidence scores

📊 Trained using Random Forest Classifier (scikit-learn)

🧹 Includes data cleaning, feature engineering, and missing-value handling

🧠 Converts raw symptoms into ML-ready vectors using a custom mapping engine

⚡ Real-time prediction using a Flask backend

👨‍⚕️ Provides doctor recommendations for predicted diseases

📁 Handles patient input and stores prediction results

🏗️ Tech Stack

Machine Learning: Python, scikit-learn, NumPy, Pandas

Model: Random Forest Classifier

Backend: Flask

Frontend: HTML, CSS

Utilities: Symptom mapping engine, JSON/CSV datasets

📂 Project Structure
Disease-Prediction-System/
│── app.py                    # Flask app
│── model.pkl                 # Trained ML model
│── mapping.py                # Symptom-to-feature mapping engine
│── templates/
│   └── index.html            # Web UI
│── static/
│   └── styles.css            # Styling
│── dataset.csv               # Medical dataset
│── README.md                 # Documentation
