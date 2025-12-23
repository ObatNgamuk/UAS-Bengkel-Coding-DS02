import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model & Scaler ---
model = joblib.load('model_churn_terbaik.pkl')
scaler = joblib.load('scaler.pkl')

# --- 2. Judul ---
st.title("Aplikasi Prediksi Customer Churn")
st.write("UAS Bengkel Koding Data Science - Prediksi Pelanggan Telecom")

# --- 3. Sidebar Input ---
st.sidebar.header("Data Pelanggan")

def user_input_features():
    # A. Input Numerik
    tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan (USD)', min_value=0.0, value=70.0)
    total_charges = st.sidebar.number_input('Total Biaya (USD)', min_value=0.0, value=500.0)
    
    # B. Input Kategorikal Utama
    contract = st.sidebar.selectbox('Contract (Jenis Kontrak)', ('Month-to-month', 'One year', 'Two year'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    # C. Input Pendukung
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    
    # --- MAPPING DATA (Sesuai X_train) ---
    data = {
        # 1. Numerik
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        
        # 2. Label Encoded (Biner)
        'gender': 1,           # Default Male
        'SeniorCitizen': 0,    # Default No
        'Partner': 0,          # Default No
        'Dependents': 0,       # Default No
        'PhoneService': 1,     # Default Yes
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        
        # 3. One-Hot Encoded (Manual Logic)
        
        # InternetService (Faktor Penting Churn!)
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        
        # PaymentMethod (Electronic Check = Resiko Tinggi)
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        
        # Contract
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        
        # Layanan Lainnya
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        
        # Fitur Sisanya (Kita set default 0/No agar simpel)
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'OnlineBackup_No internet service': 1 if internet_service == 'No' else 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 1 if internet_service == 'No' else 0,
        'DeviceProtection_Yes': 0,
        'StreamingTV_No internet service': 1 if internet_service == 'No' else 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 1 if internet_service == 'No' else 0,
        'StreamingMovies_Yes': 0,
    }
    
    # Kembalikan dataframe DAN nilai raw untuk display
    features = pd.DataFrame(data, index=[0])
    return features, contract, internet_service, payment_method

# --- 4. Tampilkan Input & Prediksi ---
input_df, raw_contract, raw_internet, raw_payment = user_input_features()

# Layout Kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil Pelanggan")
    st.info(f"**Kontrak:** {raw_contract}")
    st.info(f"**Internet:** {raw_internet}")
    st.info(f"**Pembayaran:** {raw_payment}")

with col2:
    st.subheader("Prediksi")
    if st.button('Mulai Prediksi'):
        try:
            # A. Scaling
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            # B. Reorder Columns (Wajib)
            if hasattr(model, 'feature_names_in_'):
                input_df = input_df[model.feature_names_in_]
            
            # C. Prediksi
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            churn_prob = probability[0][1] * 100 # Ambil probabilitas kelas 1 (Churn)
            
            # Tampilkan Hasil dengan Gauge/Persentase
            st.metric("Risiko Churn", f"{churn_prob:.2f}%")
            
            if prediction[0] == 1:
                st.error('⚠️ PREDIKSI: BERPOTENSI CHURN')
            else:
                st.success('✅ PREDIKSI: PELANGGAN AMAN')
                
        except Exception as e:
            st.error("Terjadi Kesalahan:")
            st.text(e)