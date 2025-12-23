import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model & Scaler (Tugas Deployment Poin 3) ---
model = joblib.load('model_churn_terbaik.pkl')
scaler = joblib.load('scaler.pkl')

# --- 2. Judul & Deskripsi ---
st.title("Aplikasi Prediksi Customer Churn")
st.write("UAS Bengkel Koding Data Science - Prediksi apakah pelanggan akan berhenti berlangganan.")

# --- 3. Form Input Fitur (Tugas Deployment Poin 3.b) ---
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    # Input Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (bulan)', min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan (USD)', min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input('Total Biaya (USD)', min_value=0.0, value=500.0)
    
    # Input Kategorikal (Penting: Harus sama dengan format training)
    # Kita hanya ambil fitur utama untuk demo ini
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Tampilkan data input user
st.subheader('Data Pelanggan yang Dimasukkan:')
st.write(input_df)


df_pred = input_df.copy()

# Mapping Manual (Label Encoding Sederhana untuk biner)
binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0}
df_pred['PaperlessBilling'] = df_pred['PaperlessBilling'].map(binary_map)

# One-Hot Encoding Manual (Sesuai kolom yang dihasilkan get_dummies di Colab)
# Misal: Contract_One year, Contract_Two year (Month-to-month biasanya didrop/jadi base)
# Agar aplikasi tidak crash, kita harus pastikan inputnya numerik semua

# Strategi Aman: Kita Scaling dulu data numeriknya
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_pred[num_cols] = scaler.transform(df_pred[num_cols])

if st.button('Prediksi Churn'):
    try:

        
        prediction = model.predict(df_pred) # Ini akan jalan jika kolomnya pas
        
        if prediction[0] == 1:
            st.error('PREDIKSI: Pelanggan Berpotensi CHURN (Berhenti).')
        else:
            st.success('PREDIKSI: Pelanggan Aman (Tidak Churn).')
            
    except Exception as e:
        st.warning("Catatan untuk Demo:")
        st.write("Model dilatih dengan puluhan kolom hasil One-Hot Encoding.")
        st.write("Untuk aplikasi demo ini, pastikan struktur kolom input sama persis dengan X_train.")
        st.error(f"Error Detail: {e}")

# Keterangan Tambahan
st.write("---")
st.write("Model Machine Learning: Voting Classifier / Random Forest")