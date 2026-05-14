import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ── Dataset: symptoms → disease ──────────────────────────────
data = {
    'fever': [1,1,1,0,0,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0],
    'cough': [1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0],
    'headache': [1,0,1,1,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0],
    'fatigue': [1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0],
    'nausea': [0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,0,1,1],
    'body_pain': [1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0],
    'sore_throat': [0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0],
    'runny_nose': [0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0],
    'chest_pain': [0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1],
    'shortness_of_breath': [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    'loss_of_taste': [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
    'diarrhea': [0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0],
    'disease': [
        'Flu','Common Cold','Migraine','Common Cold','Gastroenteritis',
        'COVID-19','Flu','Common Cold','Gastroenteritis','Flu',
        'COVID-19','Gastroenteritis','Flu','Common Cold','Gastroenteritis',
        'COVID-19','Flu','Common Cold','Gastroenteritis','Gastroenteritis'
    ]
}

df = pd.DataFrame(data)

X = df.drop('disease', axis=1)
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds)*100:.1f}%")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")