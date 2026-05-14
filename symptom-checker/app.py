import os
import pickle
import streamlit as st
import numpy as np

# --- Page config ---
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

# --- Load model ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.pkl")

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()
# ── Disease info ─────────────────────────────────────────────
disease_info = {
    'Flu': {
        'description': 'Influenza is a viral infection that attacks your respiratory system.',
        'remedies': [
            'Rest and sleep as much as possible',
            'Stay hydrated — drink water, herbal tea, or soup',
            'Take paracetamol for fever and body pain',
            'Use a humidifier to ease breathing',
        ],
        'warning': 'See a doctor if fever exceeds 103°F or symptoms worsen after 7 days.',
        'color': 'orange'
    },
    'Common Cold': {
        'description': 'A viral infection of the nose and throat, usually harmless.',
        'remedies': [
            'Drink plenty of warm fluids',
            'Gargle with salt water for sore throat',
            'Use nasal drops for congestion',
            'Rest and avoid cold environments',
        ],
        'warning': 'See a doctor if symptoms last more than 10 days.',
        'color': 'blue'
    },
    'COVID-19': {
        'description': 'A respiratory illness caused by the SARS-CoV-2 virus.',
        'remedies': [
            'Isolate immediately to prevent spreading',
            'Monitor oxygen levels with a pulse oximeter',
            'Rest and stay well hydrated',
            'Take paracetamol for fever — avoid ibuprofen',
        ],
        'warning': 'Seek emergency care if oxygen drops below 94% or breathing becomes difficult.',
        'color': 'red'
    },
    'Migraine': {
        'description': 'A neurological condition causing intense throbbing headaches.',
        'remedies': [
            'Rest in a quiet, dark room',
            'Apply cold or warm compress on forehead',
            'Stay hydrated and avoid skipping meals',
            'Take prescribed migraine medication',
        ],
        'warning': 'See a doctor if headaches are frequent or getting progressively worse.',
        'color': 'purple'
    },
    'Gastroenteritis': {
        'description': 'Inflammation of the stomach and intestines, often from infection.',
        'remedies': [
            'Drink ORS (oral rehydration solution) to prevent dehydration',
            'Eat bland foods — rice, toast, bananas',
            'Avoid dairy, fatty, or spicy food',
            'Rest and avoid strenuous activity',
        ],
        'warning': 'See a doctor if vomiting or diarrhea lasts more than 48 hours.',
        'color': 'green'
    }
}

# ── UI ───────────────────────────────────────────────────────
st.title("🩺 AI Medical Symptom Checker")
st.markdown(
    "Select your symptoms below and our AI will suggest a possible condition "
    "along with home remedies."
)
st.warning(
    "⚠️ This app is for educational purposes only. "
    "Always consult a qualified doctor for medical advice."
)

st.divider()
st.subheader("Select your symptoms")

col1, col2 = st.columns(2)

with col1:
    fever             = st.checkbox("🌡️ Fever")
    cough             = st.checkbox("😮‍💨 Cough")
    headache          = st.checkbox("🤕 Headache")
    fatigue           = st.checkbox("😴 Fatigue")
    nausea            = st.checkbox("🤢 Nausea")
    body_pain         = st.checkbox("💪 Body Pain")

with col2:
    sore_throat       = st.checkbox("🤧 Sore Throat")
    runny_nose        = st.checkbox("👃 Runny Nose")
    chest_pain        = st.checkbox("💔 Chest Pain")
    shortness_breath  = st.checkbox("😮 Shortness of Breath")
    loss_of_taste     = st.checkbox("👅 Loss of Taste/Smell")
    diarrhea          = st.checkbox("🚽 Diarrhea")

st.divider()

if st.button("🔍 Check Symptoms", use_container_width=True):

    symptoms = [
        int(fever), int(cough), int(headache), int(fatigue),
        int(nausea), int(body_pain), int(sore_throat), int(runny_nose),
        int(chest_pain), int(shortness_breath), int(loss_of_taste), int(diarrhea)
    ]

    if sum(symptoms) == 0:
        st.error("Please select at least one symptom.")
    else:
        input_data = np.array(symptoms).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probas     = model.predict_proba(input_data)[0]
        confidence = round(max(probas) * 100, 1)

        info = disease_info[prediction]

        st.divider()
        st.subheader(f"Possible condition: {prediction}")
        st.progress(int(confidence), text=f"Confidence: {confidence}%")

        st.info(info['description'])

        st.subheader("Recommended home remedies")
        for remedy in info['remedies']:
            st.markdown(f"- {remedy}")

        st.error(f"**When to see a doctor:** {info['warning']}")

        st.divider()
        st.caption(
            "This prediction is based on a machine learning model trained "
            "on symptom patterns. It is not a substitute for professional "
            "medical diagnosis."
        )