import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model & Scaler ---
model = joblib.load('model_churn_terbaik.pkl')
scaler = joblib.load('scaler.pkl')

# --- 2. Judul ---
st.title("Aplikasi Prediksi Customer Churn")
st.write("UAS Bengkel Koding Data Science - Prediksi apakah pelanggan akan berhenti berlangganan.")

# --- 3. Sidebar Input ---
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    # Input Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (bulan)', min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan (USD)', min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input('Total Biaya (USD)', min_value=0.0, value=500.0)
    
    # Input Kategorikal
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    
    # --- MAPPING MANUAL ---
    data = {
        # 1. Kolom Numerik
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        
        # 2. Kolom Label Encoded (Sesuai Training)
        'gender': 1,           # Default Male
        'SeniorCitizen': 0,    # Default No
        'Partner': 0,          # Default No
        'Dependents': 0,       # Default No
        'PhoneService': 1,     # Default Yes
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        
        # 3. Kolom One-Hot Encoded (Sesuai Training)
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        
        'OnlineBackup_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineBackup_Yes': 0,
        
        'DeviceProtection_No internet service': 1 if online_security == 'No internet service' else 0,
        'DeviceProtection_Yes': 0,
        
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        
        'StreamingTV_No internet service': 1 if online_security == 'No internet service' else 0,
        'StreamingTV_Yes': 0,
        
        'StreamingMovies_No internet service': 1 if online_security == 'No internet service' else 0,
        'StreamingMovies_Yes': 0,
        
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0
    }
    
    return pd.DataFrame(data, index=[0])

# --- 4. Proses Input & Prediksi ---
input_df = user_input_features()

st.subheader('Data Input User:')
st.write(input_df)

if st.button('Prediksi Churn'):
    try:
        # A. Scaling (Hanya kolom numerik)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # B. MENYUSUN ULANG KOLOM (SOLUSI ERROR)
        # Kita paksa urutan kolom input_df agar sama persis dengan urutan model saat training
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df[model.feature_names_in_]
        
        # C. Prediksi
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error('PREDIKSI: Pelanggan Berpotensi CHURN (Berhenti).')
        else:
            st.success('PREDIKSI: Pelanggan Aman (Tidak Churn).')
            
    except Exception as e:
        st.error("Terjadi Kesalahan:")
        st.text(e)