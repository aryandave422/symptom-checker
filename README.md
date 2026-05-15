# AI Medical Symptom Checker

An end-to-end AI web app that predicts possible medical conditions
based on symptoms entered by the user.

## Live Demo
[Click here to try the app](https://symptom-checker-3gkdu9bhulasyqwglderxa.streamlit.app/)

## Features
- 12 symptoms as input
- Predicts 5 conditions: Flu, Cold, COVID-19, Migraine, Gastroenteritis
- Shows home remedies and doctor visit warnings
- Confidence score for each prediction

## Tech Stack
- Python, scikit-learn (Random Forest)
- Streamlit for web interface
- Deployed on Streamlit Community Cloud

## Disclaimer
This app is for educational purposes only.
Always consult a qualified doctor.

## How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
