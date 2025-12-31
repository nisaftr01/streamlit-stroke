import pickle
import streamlit as st
import numpy as np
import pandas as pd

# ==============================================================================
# 1. LOAD MODEL DAN SCALER
# ==============================================================================
try:
    model = pickle.load(open('stroke_stacking_model.sav', 'rb'))
    scaler = pickle.load(open('scaler_stroke.sav', 'rb'))
except FileNotFoundError:
    st.error("File model/scaler tidak ditemukan. Pastikan file .sav ada di folder yang sama.")
    st.stop()

# ==============================================================================
# 2. JUDUL
# ==============================================================================
st.title('Prediksi Risiko Stroke (Ensemble Stacking)')
st.write("Silakan isi data pasien di bawah ini:")

# ==============================================================================
# 3. INPUT DATA (MENGGUNAKAN SELECTBOX UNTUK KATEGORI)
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    # Input Angka
    age = st.number_input('Umur Pasien', min_value=0, max_value=120, value=30)
    avg_glucose_level = st.number_input('Rata-rata Kadar Glukosa', min_value=0.0, value=100.0)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, value=25.0)
    
    # Input Pilihan (Binary 0/1)
    hypertension = st.selectbox('Apakah Hipertensi?', ['Tidak (0)', 'Ya (1)'])
    heart_disease = st.selectbox('Apakah Ada Sakit Jantung?', ['Tidak (0)', 'Ya (1)'])

with col2:
    # Input Pilihan Kategori (Teks)
    # Mapping manual ini disesuaikan dengan hasil LabelEncoder di Jupyter tadi
    
    gender_opt = st.selectbox('Jenis Kelamin', ['Female', 'Male', 'Other'])
    married_opt = st.selectbox('Pernah Menikah?', ['No', 'Yes'])
    work_opt = st.selectbox('Tipe Pekerjaan', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
    residence_opt = st.selectbox('Tipe Tempat Tinggal', ['Rural', 'Urban'])
    smoke_opt = st.selectbox('Status Merokok', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# ==============================================================================
# 4. PRE-PROCESSING (UBAH INPUT USER JADI ANGKA MESIN)
# ==============================================================================

# Helper function untuk mengubah pilihan user menjadi angka
# Nilai ini HARUS SAMA dengan urutan alfabet (karena LabelEncoder mengurutkan abjad)

# Gender: Female=0, Male=1, Other=2
gender_val = 0 if gender_opt == 'Female' else (1 if gender_opt == 'Male' else 2)

# Married: No=0, Yes=1
married_val = 0 if married_opt == 'No' else 1

# Work Type: Govt_job=0, Never_worked=1, Private=2, Self-employed=3, children=4
if work_opt == 'Govt_job': work_val = 0
elif work_opt == 'Never_worked': work_val = 1
elif work_opt == 'Private': work_val = 2
elif work_opt == 'Self-employed': work_val = 3
else: work_val = 4

# Residence: Rural=0, Urban=1
residence_val = 0 if residence_opt == 'Rural' else 1

# Smoking: Unknown=0, formerly smoked=1, never smoked=2, smokes=3
if smoke_opt == 'Unknown': smoke_val = 0
elif smoke_opt == 'formerly smoked': smoke_val = 1
elif smoke_opt == 'never smoked': smoke_val = 2
else: smoke_val = 3

# Hipertensi & Jantung (Ambil angka depannya saja dari string pilihan)
hypertension_val = 0 if 'Tidak' in hypertension else 1
heart_disease_val = 0 if 'Tidak' in heart_disease else 1

# ==============================================================================
# 5. PREDIKSI
# ==============================================================================
if st.button('Cek Risiko Stroke'):
    try:
        # Urutan kolom HARUS SAMA PERSIS dengan dataset CSV:
        # gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
        
        input_data = [
            gender_val, 
            float(age), 
            hypertension_val, 
            heart_disease_val, 
            married_val, 
            work_val, 
            residence_val, 
            float(avg_glucose_level), 
            float(bmi), 
            smoke_val
        ]
        
        # Ubah ke numpy array & Reshape
        input_np = np.array(input_data).reshape(1, -1)
        
        # SCALING (PENTING!)
        std_data = scaler.transform(input_np)
        
        # Prediksi
        prediction = model.predict(std_data)
        
        if prediction[0] == 1:
            st.error('HASIL: Pasien Berisiko Tinggi Terkena STROKE')
            st.write("Segera konsultasikan dengan dokter.")
        else:
            st.success('HASIL: Pasien Aman (Resiko Rendah)')
            st.balloons()
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")