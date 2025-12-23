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

# Kita buat input untuk fitur-fitur UTAMA yang berpengaruh
# (Sisanya kita set default agar tidak error)
def user_input_features():
    # Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (bulan)', min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan (USD)', min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input('Total Biaya (USD)', min_value=0.0, value=500.0)
    
    # Kategorikal
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    
    # --- PENTING: MAPPING MANUAL (Hardcoding) ---
    # Kita harus membuat dictionary yang strukturnya SAMA PERSIS dengan X_train
    # Urutan kolom ini berdasarkan hasil pd.get_dummies(drop_first=True) standar
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        
        # Fitur Kategorikal yang di-One-Hot Encode (Manual Logic)
        # Jika user pilih "Yes", nilainya 1. Jika "No", nilainya 0.
        
        # SeniorCitizen (Default 0/Tidak Lansia utk demo ini)
        'SeniorCitizen': 0, 
        
        # Partner & Dependents (Default No utk demo)
        'Partner': 0,
        'Dependents': 0,
        
        # PhoneService (Default Yes)
        'PhoneService': 1,
        
        # MultipleLines
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0, # Asumsi No
        
        # InternetService (Asumsi DSL sebagai default jika tidak ditanya)
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        
        # OnlineSecurity
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        
        # OnlineBackup (Default No)
        'OnlineBackup_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineBackup_Yes': 0,
        
        # DeviceProtection (Default No)
        'DeviceProtection_No internet service': 1 if online_security == 'No internet service' else 0,
        'DeviceProtection_Yes': 0,
        
        # TechSupport
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        
        # StreamingTV & Movies (Default No)
        'StreamingTV_No internet service': 1 if online_security == 'No internet service' else 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 1 if online_security == 'No internet service' else 0,
        'StreamingMovies_Yes': 0,
        
        # Contract
        # Base/Reference category adalah 'Month-to-month' (semua 0)
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        
        # PaperlessBilling
        'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
        
        # PaymentMethod (Default Electronic check)
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0
        
        # Gender (Default Male/0 karena drop_first biasanya buang Female atau Male)
        # Kita asumsikan kolomnya gender_Male. (Default 1=Male)
        # Jika error kolom kurang, tambahkan 'gender_Male': 1
    }
    
    # Tambahan handling jika ada kolom gender hasil training
    # Coba tambahkan gender_Male default 1
    data['gender_Male'] = 1 
    
    return pd.DataFrame(data, index=[0])

# --- 4. Tampilkan Input ---
input_df = user_input_features()
st.subheader('Data Input User (Setelah diproses):')
st.write(input_df)

# --- 5. Preprocessing Scaling ---
# Kita harus melakukan scaling pada kolom numerik saja
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Gunakan try-except agar aplikasi tidak crash jika ada ketidakcocokan minor
if st.button('Prediksi Churn'):
    try:
        # Scale data
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Prediksi
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] # Ambil probabilitas Churn
        
        st.write(f"Probabilitas Churn: {probability:.2%}")
        
        if prediction[0] == 1:
            st.error('PREDIKSI: Pelanggan Berpotensi CHURN (Berhenti).')
        else:
            st.success('PREDIKSI: Pelanggan Aman (Tidak Churn).')
            
    except Exception as e:
        st.error("Terjadi Kesalahan pada Dimensi Fitur.")
        st.warning("Pastikan jumlah kolom pada `data` di app.py sama persis dengan X_train saat training.")
        st.code(f"Error Detail: {e}")