import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌤️")

@st.cache_resource
def latih_model():
    try:
        df = pd.read_csv('cuaca_kemayoran_bmkg_1997_2023.csv', sep=';')
        except FileNotFoundError:
        st.error("❌ File CSV tidak ditemukan! Pastikan file 'cuaca_kemayoran_bmkg_1997_2023.csv' ada di GitHub sejajar dengan file script ini.")
        return None

    cols_numeric = [
        'Temperatur minimum(°C)', 'Temperatur maksimum(°C)',
        'Temperatur ratarata(°C)', 'Kelembapan ratarata(%)',
        'Curah hujan(mm)', 'Lamanya penyinaran matahari(jam)',
        'Kecepatan angin maksimum(m/s)', 'Kecepatan angin ratarata(m/s)'
    ]
    for col in cols_numeric:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = df[col].replace(['-', ' - '], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Curah hujan(mm)'] = df['Curah hujan(mm)'].replace([8888, 9999], np.nan)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('TANGGAL').reset_index(drop=True)
    df[cols_numeric] = df[cols_numeric].interpolate(method='linear')
    df = df.dropna().reset_index(drop=True)

    df['Bulan'] = df['TANGGAL'].dt.month
    df['Bulan_Sin'] = np.sin(2 * np.pi * df['Bulan']/12)
    df['Bulan_Cos'] = np.cos(2 * np.pi * df['Bulan']/12)

    df['Suhu_Maks_Kemarin'] = df['Temperatur maksimum(°C)'].shift(1)
    df['Hujan_Kemarin'] = df['Curah hujan(mm)'].shift(1)
    df['Lembap_Kemarin'] = df['Kelembapan ratarata(%)'].shift(1)
    df['Angin_Kemarin'] = df['Kecepatan angin maksimum(m/s)'].shift(1)

    df['Rata2_Suhu_Maks_7Hari'] = df['Temperatur maksimum(°C)'].shift(1).rolling(7).mean()
    df['Rata2_Hujan_7Hari'] = df['Curah hujan(mm)'].shift(1).rolling(7).mean()
    df['Rata2_Lembap_7Hari'] = df['Kelembapan ratarata(%)'].shift(1).rolling(7).mean()
    df['Rata2_Angin_3Hari'] = df['Kecepatan angin maksimum(m/s)'].shift(1).rolling(3).mean()

    df['Target_Suhu_Maks'] = df['Temperatur maksimum(°C)'].shift(-1)
    df['Target_Status_Hujan'] = (df['Curah hujan(mm)'].shift(-1) > 1.0).astype(int)
    df['Target_Hujan_Amount'] = df['Curah hujan(mm)'].shift(-1)
    df['Target_Lembap'] = df['Kelembapan ratarata(%)'].shift(-1)
    df['Target_Angin'] = df['Kecepatan angin maksimum(m/s)'].shift(-1)

    df_model = df.dropna().reset_index(drop=True)

    feats_suhu = ['Suhu_Maks_Kemarin', 'Rata2_Suhu_Maks_7Hari', 'Lembap_Kemarin', 'Hujan_Kemarin', 'Bulan_Sin', 'Bulan_Cos']
    scaler_suhu = StandardScaler()
    X_suhu_scaled = scaler_suhu.fit_transform(df_model[feats_suhu])
    rf_suhu = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_suhu.fit(X_suhu_scaled, df_model['Target_Suhu_Maks'])

    feats_hujan = ['Hujan_Kemarin', 'Rata2_Hujan_7Hari', 'Lembap_Kemarin', 'Suhu_Maks_Kemarin', 'Bulan_Sin', 'Bulan_Cos']
    scaler_hujan = StandardScaler()
    scaler_hujan.fit(df_model[feats_hujan]) 
    
    rf_class = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_class.fit(df_model[feats_hujan], df_model['Target_Status_Hujan'])
    
    mask_rain = df_model['Target_Status_Hujan'] == 1
    rf_amt_hujan_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf_amt_hujan_model.fit(df_model.loc[mask_rain, feats_hujan], df_model.loc[mask_rain, 'Target_Hujan_Amount'])

    feats_hum = ['Lembap_Kemarin', 'Rata2_Lembap_7Hari', 'Hujan_Kemarin', 'Suhu_Maks_Kemarin', 'Bulan_Sin', 'Bulan_Cos']
    scaler_hum = StandardScaler()
    X_hum_scaled = scaler_hum.fit_transform(df_model[feats_hum])
    rf_hum = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_hum.fit(X_hum_scaled, df_model['Target_Lembap'])

    feats_wind = ['Angin_Kemarin', 'Rata2_Angin_3Hari', 'Suhu_Maks_Kemarin', 'Hujan_Kemarin', 'Bulan_Sin', 'Bulan_Cos']
    scaler_wind = StandardScaler()
    X_wind_scaled = scaler_wind.fit_transform(df_model[feats_wind])
    rf_wind = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_wind.fit(X_wind_scaled, df_model['Target_Angin'])

    return (rf_suhu, scaler_suhu, feats_suhu), \
           (rf_class, rf_amt_hujan_model, scaler_hujan, feats_hujan), \
           (rf_hum, scaler_hum, feats_hum), \
           (rf_wind, scaler_wind, feats_wind)

st.title("🌤️ Dashboard Prediksi Cuaca")
st.write("Aplikasi ini memprediksi cuaca besok berdasarkan data hari ini.")

with st.spinner('Sedang memuat model data...'):
    models = latih_model()

if models:
    (m_suhu, sc_suhu, f_suhu), (m_cls_hujan, m_reg_hujan, sc_hujan, f_hujan), (m_hum, sc_hum, f_hum), (m_wind, sc_wind, f_wind) = models

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Data Hari Ini")
        tgl_besok = st.date_input("Tanggal yang ingin diprediksi")
        suhu_kemarin = st.number_input("Suhu Maks Hari Ini (°C)", value=33.0)
        hujan_kemarin = st.number_input("Curah Hujan Hari Ini (mm)", value=0.0)
        lembap_kemarin = st.number_input("Kelembapan Rata-rata (%)", value=75.0)
        angin_kemarin = st.number_input("Kecepatan Angin (m/s)", value=5.0)

    with col2:
        st.subheader("2. Data Tren (Rata-rata)")
        st.info("Jika tidak tahu, biarkan nilai default (sama dengan hari ini).")
        avg_suhu_7 = st.number_input("Rata2 Suhu (7 hari lalu)", value=suhu_kemarin)
        avg_hujan_7 = st.number_input("Rata2 Hujan (7 hari lalu)", value=hujan_kemarin)
        avg_lembap_7 = st.number_input("Rata2 Lembap (7 hari lalu)", value=lembap_kemarin)
        avg_angin_3 = st.number_input("Rata2 Angin (3 hari lalu)", value=angin_kemarin)

    if st.button("🔍 Mulai Prediksi", type="primary"):

        date_obj = pd.to_datetime(tgl_besok)
        bulan = date_obj.month
        bulan_sin = np.sin(2 * np.pi * bulan/12)
        bulan_cos = np.cos(2 * np.pi * bulan/12)

        input_suhu = pd.DataFrame([[suhu_kemarin, avg_suhu_7, lembap_kemarin, hujan_kemarin, bulan_sin, bulan_cos]], columns=f_suhu)
        pred_suhu_val = m_suhu.predict(sc_suhu.transform(input_suhu))[0]

        input_hum = pd.DataFrame([[lembap_kemarin, avg_lembap_7, hujan_kemarin, suhu_kemarin, bulan_sin, bulan_cos]], columns=f_hum)
        pred_hum_val = m_hum.predict(sc_hum.transform(input_hum))[0]

        input_wind = pd.DataFrame([[angin_kemarin, avg_angin_3, suhu_kemarin, hujan_kemarin, bulan_sin, bulan_cos]], columns=f_wind)
        pred_wind_val = m_wind.predict(sc_wind.transform(input_wind))[0]

        input_hujan = pd.DataFrame([[hujan_kemarin, avg_hujan_7, lembap_kemarin, suhu_kemarin, bulan_sin, bulan_cos]], columns=f_hujan)
        prob_hujan = m_cls_hujan.predict_proba(input_hujan)[0][1]
        
        status_hujan = "HUJAN" if prob_hujan > 0.4 else "TIDAK HUJAN"
        raw_rain_amt = m_reg_hujan.predict(input_hujan)[0]
        final_rain_amt = raw_rain_amt if prob_hujan > 0.4 else 0.0

        st.success("Prediksi Selesai!")
        st.header(f"📅 Prediksi: {tgl_besok.strftime('%d-%m-%Y')}")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Suhu Maks", f"{pred_suhu_val:.1f} °C")
        col_res2.metric("Kelembapan", f"{pred_hum_val:.1f} %")
        col_res3.metric("Angin", f"{pred_wind_val:.1f} m/s")
        col_res4.metric("Curah Hujan", f"{final_rain_amt:.1f} mm")
        
        st.subheader(f"Status Cuaca: {status_hujan}")
        st.progress(int(prob_hujan*100), text=f"Probabilitas Hujan: {prob_hujan*100:.1f}%")
