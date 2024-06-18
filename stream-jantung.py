import streamlit as st
import numpy as np
import pickle

# Memuat model yang sudah disimpan
try:
    with open('naive_bayes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.write(f"Error loading the model: {e}")
    st.stop()

# Judul aplikasi
st.title('Prediksi Risiko Penyakit Jantung')

# Input pengguna
age = st.number_input('Usia', min_value=1, max_value=120, value=40)
resting_bp_s = st.number_input('Tekanan Darah', min_value=50, max_value=200, value=120)
cholesterol = st.number_input('Kolesterol', min_value=100, max_value=400, value=200)
max_heart_rate = st.number_input('Detak Jantung Maksimum', min_value=60, max_value=220, value=150)
oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox('ST Slope', [0, 1, 2], format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])

sex_encoded = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: ['Laki-laki', 'Perempuan'][x])
chest_pain_type_encoded = st.selectbox(
    'Tipe Nyeri Dada', 
    [0, 1, 2, 3], 
    format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x]
)
fasting_blood_sugar_encoded = st.selectbox(
    'Gula Darah Puasa > 120 mg/dl', 
    [0, 1], 
    format_func=lambda x: ['False', 'True'][x]
)
resting_ecg_encoded = st.selectbox(
    'Hasil Elektrokardiografi', 
    [0, 1], 
    format_func=lambda x: ['Normal', 'Abnormal'][x]
)
exercise_angina_encoded = st.selectbox(
    'Nyeri dada saat olahraga', 
    [0, 1], 
    format_func=lambda x: ['No', 'Yes'][x]
)

# Membuat fitur tambahan dari chest_pain_type_encoded
chest_pain_type_encoded_1 = 1 if chest_pain_type_encoded == 1 else 0
chest_pain_type_encoded_2 = 1 if chest_pain_type_encoded == 2 else 0
chest_pain_type_encoded_3 = 1 if chest_pain_type_encoded == 3 else 0

# Membuat array dari input pengguna
input_data = np.array([[age, resting_bp_s, cholesterol, max_heart_rate, oldpeak, st_slope, 
                        sex_encoded, chest_pain_type_encoded, fasting_blood_sugar_encoded, 
                        resting_ecg_encoded, exercise_angina_encoded,
                        chest_pain_type_encoded_1, chest_pain_type_encoded_2, chest_pain_type_encoded_3]])

# Membuat prediksi
if st.button('Prediksi'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('Anda berisiko terkena penyakit jantung.')
    else:
        st.write('Anda tidak berisiko terkena penyakit jantung.')
