import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Data Input (Fitur)': feature_names,
        'Tingkat Pengaruh (%)': (importances * 100).round(2)
    })
    df_imp = df_imp.sort_values('Tingkat Pengaruh (%)', ascending=False).reset_index(drop=True)
    print(f"--- MODEL: {model_name} ---")
    print(f"Faktor Utama: {df_imp.iloc[0]['Data Input (Fitur)']} ({df_imp.iloc[0]['Tingkat Pengaruh (%)']} %)")
    print(df_imp.to_string(index=False))
    print("-" * 60 + "\n")

# Data Preparation
df = pd.read_csv('C:/Kuliah/Semester 3/(Datnal) Data Analyst/W13/[Unguided]_Benedictus_Darrell_Sunanto_Arup_00000118095_Data_Analyst_W13/cuaca_kemayoran_bmkg_1997_2023.csv', sep=';')

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

# Feature Engineering
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
split_idx = int(len(df_model) * 0.8)

# Model Suhu Maksimum
feats_suhu = ['Suhu_Maks_Kemarin', 'Rata2_Suhu_Maks_7Hari', 'Lembap_Kemarin',
            'Hujan_Kemarin', 'Bulan_Sin', 'Bulan_Cos']

X_suhu = df_model[feats_suhu]
y_suhu = df_model['Target_Suhu_Maks']

X_train_suhu, X_test_suhu, y_train_suhu, y_test_suhu = train_test_split(X_suhu, y_suhu, test_size=0.2, shuffle=False, random_state=42)

scaler_suhu = StandardScaler()
X_train_scaled_suhu = scaler_suhu.fit_transform(X_train_suhu)
X_test_scaled_suhu = scaler_suhu.transform(X_test_suhu)

rf_suhu = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_suhu.fit(X_train_scaled_suhu, y_train_suhu)
pred_suhu = rf_suhu.predict(X_test_scaled_suhu)

mae_suhu = mean_absolute_error(y_test_suhu, pred_suhu)
print(f"MAE Suhu Maks: {mae_suhu:.2f} °C")

# Model Hujan (Two-Stage)
feats_hujan = ['Hujan_Kemarin', 'Rata2_Hujan_7Hari', 'Lembap_Kemarin',
            'Suhu_Maks_Kemarin', 'Bulan_Sin', 'Bulan_Cos']

X_hujan = df_model[feats_hujan]
y_stat = df_model['Target_Status_Hujan']
y_amt_hujan = df_model['Target_Hujan_Amount']

X_train_hujan, X_test_hujan, y_train_stat, y_test_stat, y_train_amt_hujan, y_test_amt_hujan = train_test_split(
    X_hujan, y_stat, y_amt_hujan, test_size=0.2, shuffle=False, random_state=42
)

scaler_hujan = StandardScaler()
X_train_scaled_hujan = scaler_hujan.fit_transform(X_train_hujan)
X_test_scaled_hujan = scaler_hujan.transform(X_test_hujan)

# Classification
rf_class = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf_class.fit(X_train_hujan, y_train_stat)
pred_stat = rf_class.predict(X_test_hujan)
prob_rain = rf_class.predict_proba(X_test_hujan)[:, 1]

# Regression (only for rainy days)
mask_rain = y_train_stat == 1
rf_amt_hujan_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_amt_hujan_model.fit(X_train_hujan[mask_rain], y_train_amt_hujan[mask_rain])

# Combined Prediction
pred_raw_hujan = rf_amt_hujan_model.predict(X_test_hujan)
pred_amt_hujan_combined = np.where(prob_rain > 0.4, pred_raw_hujan, 0)

acc_hujan = accuracy_score(y_test_stat, pred_stat)
mae_hujan_combined = mean_absolute_error(y_test_amt_hujan, pred_amt_hujan_combined)
print(f"Akurasi Status: {acc_hujan*100:.1f}% | MAE Jumlah (Combined): {mae_hujan_combined:.2f} mm")


# Model Kelembapan
feats_hum = ['Lembap_Kemarin', 'Rata2_Lembap_7Hari', 'Hujan_Kemarin',
            'Suhu_Maks_Kemarin', 'Bulan_Sin', 'Bulan_Cos']

X_hum = df_model[feats_hum]
y_hum = df_model['Target_Lembap']

X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(X_hum, y_hum, test_size=0.2, shuffle=False, random_state=42)

scaler_hum = StandardScaler()
X_train_scaled_hum = scaler_hum.fit_transform(X_train_hum)
X_test_scaled_hum = scaler_hum.transform(X_test_hum)

rf_hum = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_hum.fit(X_train_scaled_hum, y_train_hum)
pred_hum = rf_hum.predict(X_test_scaled_hum)

mae_hum = mean_absolute_error(y_test_hum, pred_hum)
print(f"MAE Kelembapan: {mae_hum:.2f} %")


# Model Kecepatan Angin Maksimum
feats_wind = ['Angin_Kemarin', 'Rata2_Angin_3Hari', 'Suhu_Maks_Kemarin',
            'Hujan_Kemarin', 'Bulan_Sin', 'Bulan_Cos']

X_wind = df_model[feats_wind]
y_wind = df_model['Target_Angin']

X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X_wind, y_wind, test_size=0.2, shuffle=False, random_state=42)

scaler_wind = StandardScaler()
X_train_scaled_wind = scaler_wind.fit_transform(X_train_wind)
X_test_scaled_wind = scaler_wind.transform(X_test_wind)

rf_wind = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_wind.fit(X_train_scaled_wind, y_train_wind)
pred_wind = rf_wind.predict(X_test_scaled_wind)

mae_wind = mean_absolute_error(y_test_wind, pred_wind)
print(f"MAE Angin: {mae_wind:.2f} m/s")


# Model Curah Hujan (Amount only, not two-stage combined MAE)
feats_rain_amount = ['Hujan_Kemarin', 'Rata2_Hujan_7Hari', 'Suhu_Maks_Kemarin',
            'Lembap_Kemarin','Bulan_Sin', 'Bulan_Cos']

X_rain_amount = df_model[feats_rain_amount]
y_rain_amount = df_model['Target_Hujan_Amount']

X_train_rain_amount, X_test_rain_amount, y_train_rain_amount, y_test_rain_amount = train_test_split(
    X_rain_amount, y_rain_amount, test_size=0.2, shuffle=False, random_state=42
)

scaler_amt_rain = StandardScaler() # Create a new scaler for amount
X_train_scaled_rain_amount = scaler_amt_rain.fit_transform(X_train_rain_amount)
X_test_scaled_rain_amount = scaler_amt_rain.transform(X_test_rain_amount)

rf_amt_rain = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_amt_rain.fit(X_train_scaled_rain_amount, y_train_rain_amount)
pred_amt_rain = rf_amt_rain.predict(X_test_scaled_rain_amount)

mae_hujan_amount = mean_absolute_error(y_test_rain_amount, pred_amt_rain)
print(f"MAE Curah Hujan (Amount only): {mae_hujan_amount:.2f} mm")

# Feature Importance Analysis
check_feature_importance(rf_suhu, feats_suhu, "Suhu Maksimum")
check_feature_importance(rf_class, feats_hujan, "Status Hujan (Klasifikasi)")
check_feature_importance(rf_hum, feats_hum, "Kelembapan Udara")
check_feature_importance(rf_wind, feats_wind, "Kecepatan Angin")
check_feature_importance(rf_amt_rain, feats_rain_amount, "Jumlah Curah Hujan (mm)")

# Summary Evaluation
kualitas_suhu = 'SANGAT BAIK' if mae_suhu < 1.5 else 'CUKUP'
kualitas_stat = 'BAIK' if acc_hujan > 0.75 else 'CUKUP'
kualitas_hum = 'SANGAT BAIK' if mae_hum < 5.0 else 'CUKUP'
kualitas_wind = 'BAIK' if mae_wind < 2.0 else 'CUKUP'
kualitas_rain = 'BAIK' if mae_hujan_amount < 15.0 else 'CUKUP (SULIT)'

print("\n=== RANGKUMAN EVALUASI SEMUA MODEL YANG DIGUNAKAN ===")
summary_data = {
    'Nama Model': [
        'RF Regressor (Suhu)',
        'RF Classifier (Status)',
        'RF Regressor (Kelembapan)',
        'RF Regressor (Angin)',
        'RF Regressor (Jml Hujan)'
    ],
    'Target Prediksi': [
        'Temperatur MAKSIMAL (°C)',
        'Status Hujan (Ya/Tidak)',
        'Kelembapan Rata-rata (%)',
        'Kecepatan Angin Maks (m/s)',
        'Curah Hujan (mm)'
    ],
    'Metrik & Arti': [
        'MAE (Rata-rata Meleset)',
        'Accuracy (Ketepatan Tebakan)',
        'MAE (Rata-rata Meleset)',
        'MAE (Rata-rata Meleset)',
        'MAE (Rata-rata Meleset)'
    ],
    'Nilai Skor': [
        f"{mae_suhu:.2f} °C",
        f"{acc_hujan*100:.1f} %",
        f"{mae_hum:.2f} %",
        f"{mae_wind:.2f} m/s",
        f"{mae_hujan_amount:.2f} mm"
    ],
    'Kualitas': [
        kualitas_suhu,
        kualitas_stat,
        kualitas_hum,
        kualitas_wind,
        kualitas_rain
    ]
}
df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# --- FUNGSI BANTUAN (HELPER) ---
def input_angka_fleksibel(pesan):
    """
    Fungsi untuk meminta input angka yang aman.
    - Menerima koma (,) atau titik (.)
    - Jika user input huruf, dia akan minta ulang.
    """
    while True:
        data = input(pesan)
        data_bersih = data.replace(',', '.') # Ubah koma jadi titik
        
        try:
            nilai = float(data_bersih)
            return nilai
        except ValueError:
            print("❌ Input tidak valid! Harap masukkan angka (Contoh: 25.5 atau 25,5).")

# --- FUNGSI UTAMA ---
def prediksi_cuaca_besok():
    while True:
        print("\n" + "="*50)
        print("   PROGRAM PREDIKSI CUACA HARIAN")
        print("="*50)
        print("Instruksi: Gunakan titik (.) atau koma (,) untuk desimal.")
        print("-" * 50)

        try:
            # 1. INPUT TANGGAL
            while True:
                tgl_input = input("Masukkan Tanggal Besok (YYYY-MM-DD): ")
                try:
                    pd.to_datetime(tgl_input)
                    break
                except:
                    print("❌ Format tanggal salah. Gunakan format Tahun-Bulan-Hari (misal: 2024-12-25)")

            # 2. INPUT DATA HARIAN (WAJIB)
            print("\n--- Data Cuaca HARI INI ---")
            suhu_kemarin = input_angka_fleksibel("Suhu Maksimum Hari Ini (°C): ")
            hujan_kemarin = input_angka_fleksibel("Curah Hujan Hari Ini (mm): ")
            lembap_kemarin = input_angka_fleksibel("Kelembapan Rata-rata Hari Ini (%): ")
            angin_kemarin = input_angka_fleksibel("Kecepatan Angin Maks Hari Ini (m/s): ")
            
            # 3. OPSI INPUT DATA TREN (RATA-RATA)
            print("\n--- Opsi Data Tren (Rata-rata) ---")
            print("Apakah Anda memiliki data rata-rata 7 hari terakhir?")
            
            # Validasi input y/n
            while True:
                pilihan_tren = input("Input data rata-rata secara manual? (y/n): ").lower().strip()
                if pilihan_tren in ['y', 'n']:
                    break
                print("❌ Harap ketik 'y' untuk Ya atau 'n' untuk Tidak.")

            if pilihan_tren == 'y':
                # Jika user mau input manual
                print("\nSilakan masukkan data rata-rata:")
                avg_suhu_7 = input_angka_fleksibel("Rata-rata Suhu Maks (7 hari terakhir): ")
                avg_hujan_7 = input_angka_fleksibel("Rata-rata Curah Hujan (7 hari terakhir): ")
                avg_lembap_7 = input_angka_fleksibel("Rata-rata Kelembapan (7 hari terakhir): ")
                avg_angin_3 = input_angka_fleksibel("Rata-rata Angin Maks (3 hari terakhir): ")
            else:
                # Jika user tidak punya data, pakai data hari ini sebagai estimasi
                print("\n[Info] Menggunakan data hari ini sebagai pengganti nilai rata-rata...")
                avg_suhu_7 = suhu_kemarin
                avg_hujan_7 = hujan_kemarin
                avg_lembap_7 = lembap_kemarin
                avg_angin_3 = angin_kemarin

            # 4. FEATURE ENGINEERING
            date_obj = pd.to_datetime(tgl_input)
            bulan = date_obj.month
            bulan_sin = np.sin(2 * np.pi * bulan/12)
            bulan_cos = np.cos(2 * np.pi * bulan/12)

            # 5. MENYIAPKAN DATAFRAME
            input_suhu = pd.DataFrame([[suhu_kemarin, avg_suhu_7, lembap_kemarin, hujan_kemarin, bulan_sin, bulan_cos]], 
                                      columns=feats_suhu)
            
            input_hujan = pd.DataFrame([[hujan_kemarin, avg_hujan_7, lembap_kemarin, suhu_kemarin, bulan_sin, bulan_cos]], 
                                       columns=feats_hujan)
            
            input_hum = pd.DataFrame([[lembap_kemarin, avg_lembap_7, hujan_kemarin, suhu_kemarin, bulan_sin, bulan_cos]], 
                                     columns=feats_hum)
            
            input_wind = pd.DataFrame([[angin_kemarin, avg_angin_3, suhu_kemarin, hujan_kemarin, bulan_sin, bulan_cos]], 
                                      columns=feats_wind)
            
            input_rain_amt = pd.DataFrame([[hujan_kemarin, avg_hujan_7, suhu_kemarin, lembap_kemarin, bulan_sin, bulan_cos]],
                                          columns=feats_rain_amount)

            # 6. EKSEKUSI PREDIKSI
            pred_suhu_val = rf_suhu.predict(scaler_suhu.transform(input_suhu))[0]
            pred_hum_val = rf_hum.predict(scaler_hum.transform(input_hum))[0]
            pred_wind_val = rf_wind.predict(scaler_wind.transform(input_wind))[0]
            
            X_hujan_scaled = scaler_hujan.transform(input_hujan)
            prob_hujan = rf_class.predict_proba(X_hujan_scaled)[0][1]
            status_hujan = "HUJAN" if prob_hujan > 0.4 else "TIDAK HUJAN"
            
            raw_rain_amt = rf_amt_hujan_model.predict(input_hujan)[0]
            final_rain_amt = raw_rain_amt if prob_hujan > 0.4 else 0.0

            # 7. TAMPILKAN HASIL
            print("\n" + "="*50)
            print(f"   HASIL PREDIKSI: {tgl_input}")
            print("="*50)
            print(f"🌡️  Suhu Maksimum     : {pred_suhu_val:.2f} °C")
            print(f"💧 Kelembapan        : {pred_hum_val:.2f} %")
            print(f"💨 Kecepatan Angin   : {pred_wind_val:.2f} m/s")
            print("-" * 30)
            print(f"🌧️  Potensi Hujan     : {status_hujan} ({prob_hujan*100:.1f}%)")
            print(f"☔  Estimasi Curah    : {final_rain_amt:.2f} mm")
            print("="*50)

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Terjadi kesalahan: {e}")
            print("Silakan cek input dan coba lagi.")

        # 8. KONFIRMASI KELUAR
        lagi = input("\nApakah ingin memprediksi tanggal lain? (y/n): ").lower()
        if lagi != 'y':
            print("Terima kasih! Program selesai.")
            break

# Panggil fungsi utama
prediksi_cuaca_besok()