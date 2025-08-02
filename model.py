import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from prophet import Prophet
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import glob
from functools import reduce

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# # === Fungsi evaluasi ===
# def evaluate_model(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     non_zero = y_true != 0
#     y_true_safe = y_true[non_zero]
#     y_pred_safe = y_pred[non_zero]

#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else np.nan
#     rmspe = np.sqrt(np.mean(np.square((y_true_safe - y_pred_safe) / y_true_safe))) * 100 if len(y_true_safe) > 0 else np.nan
#     return mae, rmse, mape, rmspe


# # === Fungsi Prophet umum ===
# def predict_with_prophet(df, year_col, value_col, label, groupby_avg=True):
#     df = df[[year_col, value_col]].copy()
#     df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
#     df = df.replace(0, np.nan)
#     if groupby_avg:
#         df = df.groupby(year_col).mean().reset_index()
#     df = df.dropna()
#     df["ds"] = pd.to_datetime(df[year_col], format="%Y")
#     df = df.rename(columns={value_col: "y"})

#     model = Prophet(yearly_seasonality=False, daily_seasonality=False)
#     model.fit(df)

#     forecast_train = model.predict(df[["ds"]])
#     y_true = df["y"].values
#     y_pred = forecast_train["yhat"].values
#     mae, rmse, mape, rmspe = evaluate_model(y_true, y_pred)
#     print(f"\n📊 Prophet - {label}")
#     print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, RMSPE={rmspe:.2f}%")

#     future = model.make_future_dataframe(periods=2026 - df["ds"].dt.year.max(), freq="Y")
#     forecast = model.predict(future)
#     output = forecast[["ds", "yhat"]].copy()
#     output.columns = ["Tahun", f"Prediksi_{label}"]
#     output["Tahun"] = output["Tahun"].dt.year
#     output = output[output["Tahun"] > df[year_col].max()]
#     return output

# === Load dan bersihkan data ===
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'dataset')
all_files = glob.glob(os.path.join(data_dir, '*.csv'))
raw_dfs = [pd.read_csv(f) for f in all_files]

drop_cols = ["No", "Wilayah", "Satuan", "Keterangan", "Kode Provinsi"]
rename_prov = {
    'Kep Bangka Belitung': 'Kepulauan Bangka Belitung',
    'Kep Riau': 'Kepulauan Riau',
    'Ntb': 'Nusa Tenggara Barat',
    'Ntt': 'Nusa Tenggara Timur'
}

def clean_data(dfs):
    cleaned = []
    for df in dfs:
        if 'Provinsi' in df.columns:
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            df = df.dropna(subset=['Provinsi'])
            df['Provinsi'] = df['Provinsi'].apply(lambda x: x.title()).replace(rename_prov)
            df = df[df['Provinsi'] != 'Kota Jambi']
        cleaned.append(df)
    return cleaned

df1, df2, df3, df4, df5 = cleaned_dfs = clean_data(raw_dfs)
df_predict = reduce(lambda left, right: pd.merge(left, right, on=['Tahun', "Provinsi"], how='outer'), cleaned_dfs).fillna(0)

df_predict['Jumlah_penduduk_gizi_cukup'] = df_predict['Jumlah_Penduduk'] - df_predict['Penduduk_Undernourish']
df_predict['presentase_penduduk_kurang_gizi'] = 100 - round((df_predict['Jumlah_penduduk_gizi_cukup'] / df_predict['Jumlah_Penduduk']) * 100)
df_predict['persentase_penduduk_gizi_cukup'] = round((df_predict['Jumlah_penduduk_gizi_cukup'] / df_predict['Jumlah_Penduduk']) * 100)
df_predict = df_predict[df_predict['Jumlah_Penduduk'] > df_predict['Penduduk_Undernourish']]

# # === Jalankan prediksi ===
# ikp_future = predict_with_prophet(df_predict, "Tahun", "IKP", "indeksketahananpangan")
# pou_future = predict_with_prophet(df_predict, "Tahun", "PoU", "PoU")
# kurang_gizi_future = predict_with_prophet(df_predict, "Tahun", "presentase_penduduk_kurang_gizi", "KurangGizi")
# pph_future = predict_with_prophet(df_predict, "Tahun", "Skor PPH", "PPH")
# konsumsi_future = predict_with_prophet(df_predict, "Tahun", "Konsumsi Energi (kkal/kap/hari)", "KonsumsiEnergi")
# penduduk_future = predict_with_prophet(df_predict, "Tahun", "Jumlah_Penduduk", "JumlahPenduduk")

# # === Gabungkan hasil akhir ===
# final_df = ikp_future.merge(pou_future, on="Tahun", how="outer") \
#                      .merge(kurang_gizi_future, on="Tahun", how="outer") \
#                      .merge(pph_future, on="Tahun", how="outer") \
#                      .merge(konsumsi_future, on="Tahun", how="outer") \
#                      .merge(penduduk_future, on="Tahun", how="outer")

# final_df = final_df.sort_values("Tahun").reset_index(drop=True)
# print("\n📈 Prediksi hingga 2025:")
# print(final_df.round(2))


# ===== FUNGSI EVALUASI =====
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))) * 100
    return mae, rmse, mape, rmspe

# ===== LOAD DAN SIAPKAN DATA =====


# Gunakan data tahun 2018–2023 untuk training
df_train = df_predict[df_predict["Tahun"].between(2018, 2023)]

# ===== MODEL 1: RANDOM FOREST (Prediksi IKP) =====
features_rf = ["Tahun", "PoU", "Skor PPH", "Konsumsi Energi (kkal/kap/hari)", "presentase_penduduk_kurang_gizi"]
target_rf = "IKP"

X_rf = df_train[features_rf]
y_rf = df_train[target_rf]

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_rf, y_rf)

# Input manual untuk prediksi 2024–2025
input_manual_rf = pd.DataFrame([
    {
        "Tahun": 2024,
        "PoU": 13.23,
        "Skor PPH": 91.17,
        "Konsumsi Energi (kkal/kap/hari)": 2026.07,
        "presentase_penduduk_kurang_gizi": 13.13
    },
    {
        "Tahun": 2025,
        "PoU":   13.64 ,
        "Skor PPH":  93.20,
        "Konsumsi Energi (kkal/kap/hari)": 2012.15,
        "presentase_penduduk_kurang_gizi": 13.55
    }
])

# Evaluasi di training set
y_pred_rf_train = model_rf.predict(X_rf)
mae_rf, rmse_rf, mape_rf, rmspe_rf = evaluate_model(y_rf, y_pred_rf_train)
print("\n🌲 Random Forest - Prediksi IKP")
print(f"MAE={mae_rf:.2f}, RMSE={rmse_rf:.2f}, MAPE={mape_rf:.2f}%, RMSPE={rmspe_rf:.2f}%")

# Prediksi 2024–2025
pred_rf = model_rf.predict(input_manual_rf)
input_manual_rf["Prediksi_IKP"] = pred_rf
print("\n📈 Prediksi IKP Tahun 2024–2025:")
print(input_manual_rf[["Tahun", "Prediksi_IKP"]])

# ===== MODEL 2: GRADIENT BOOSTING (Prediksi PoU) =====
features_gb = ["Tahun", "IKP", "Skor PPH", "Konsumsi Energi (kkal/kap/hari)", "Jumlah_Penduduk"]
target_gb = "PoU"

X_gb = df_train[features_gb]
y_gb = df_train[target_gb]

model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_gb, y_gb)

# Input manual untuk prediksi 2024–2025
input_manual_gb = pd.DataFrame([
    {
        "Tahun": 2024,
        "IKP": 77.45,
        "Skor PPH": 91.17,
        "Konsumsi Energi (kkal/kap/hari)": 2026.07,
      
        "Jumlah_Penduduk": 8592403.43
    },
    {
        "Tahun": 2025,
        "IKP": 78.93,
        "Skor PPH": 93.20,
        "Konsumsi Energi (kkal/kap/hari)": 2012.15,
       
        "Jumlah_Penduduk":  8736364.91
    }
])

# Evaluasi di training set
y_pred_gb_train = model_gb.predict(X_gb)
mae_gb, rmse_gb, mape_gb, rmspe_gb = evaluate_model(y_gb, y_pred_gb_train)
print("\n🌱 Gradient Boosting - Prediksi PoU")
print(f"MAE={mae_gb:.2f}, RMSE={rmse_gb:.2f}, MAPE={mape_gb:.2f}%, RMSPE={rmspe_gb:.2f}%")

# Prediksi 2024–2025
pred_gb = model_gb.predict(input_manual_gb)
input_manual_gb["Prediksi_PoU"] = pred_gb
print("\n📈 Prediksi PoU Tahun 2024–2025:")
print(input_manual_gb[["Tahun", "Prediksi_PoU"]])


#########################

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Fungsi evaluasi model
def evaluate_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    y_true_safe = y_true[non_zero]
    y_pred_safe = y_pred[non_zero]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else np.nan
    rmspe = np.sqrt(np.mean(np.square((y_true_safe - y_pred_safe) / y_true_safe))) * 100 if len(y_true_safe) > 0 else np.nan
    return mae, rmse, mape, rmspe

# Gunakan data tahun 2018–2023 untuk training
df_train = df_predict[df_predict["Tahun"].between(2018, 2023)]

# Fitur dan target
features_xgb = ["Tahun", "IKP", "Skor PPH", "Konsumsi Energi (kkal/kap/hari)", "Jumlah_Penduduk"]
target_xgb = "PoU"

X_xgb = df_train[features_xgb]
y_xgb = df_train[target_xgb]

# Inisialisasi dan latih model
model_xgb = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_xgb, y_xgb)

# Input manual prediksi 2024–2025
input_manual_xgb = pd.DataFrame([
    {
        "Tahun": 2024,
        "IKP": 77.45,
        "Skor PPH": 91.17,
        "Konsumsi Energi (kkal/kap/hari)": 2026.07,
        "Jumlah_Penduduk": 8592403.43
    },
    {
        "Tahun": 2025,
        "IKP": 78.93,
        "Skor PPH": 93.20,
        "Konsumsi Energi (kkal/kap/hari)": 2012.15,
        "Jumlah_Penduduk": 8736364.91
    }
])

# Evaluasi di training set
y_pred_train = model_xgb.predict(X_xgb)
mae_xgb, rmse_xgb, mape_xgb, rmspe_xgb = evaluate_model(y_xgb, y_pred_train)
print("\n🚀 XGBoost - Prediksi PoU")
print(f"MAE={mae_xgb:.2f}, RMSE={rmse_xgb:.2f}, MAPE={mape_xgb:.2f}%, RMSPE={rmspe_xgb:.2f}%")

# Prediksi untuk 2024–2025
pred_xgb = model_xgb.predict(input_manual_xgb)
input_manual_xgb["Prediksi_PoU"] = pred_xgb
print("\n📈 Prediksi PoU Tahun 2024–2025 (XGBoost):")
print(input_manual_xgb[["Tahun", "Prediksi_PoU"]])


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fungsi evaluasi


def evaluate_model(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter hanya yang y_true ≠ 0
    non_zero = y_true != 0
    y_true_safe = y_true[non_zero]
    y_pred_safe = y_pred[non_zero]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Hitung MAPE & RMSPE hanya dari data yang y ≠ 0
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else np.nan
    rmspe = np.sqrt(np.mean(np.square((y_true_safe - y_pred_safe) / y_true_safe))) * 100 if len(y_true_safe) > 0 else np.nan

    return mae, rmse, mape, rmspe


# Fitur dan target untuk model XGBoost
features_xgb = ["Tahun", "PoU", "Skor PPH", "Konsumsi Energi (kkal/kap/hari)", "presentase_penduduk_kurang_gizi"]
target_xgb = "IKP"

# Data training dari tahun 2018–2023
df_train = df_predict[df_predict["Tahun"].between(2018, 2023)]
X_xgb = df_train[features_xgb]
y_xgb = df_train[target_xgb]

# Training model XGBoost
model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(X_xgb, y_xgb)

# Evaluasi di training set
y_pred_xgb_train = model_xgb.predict(X_xgb)
mae, rmse, mape, rmspe = evaluate_model(y_xgb, y_pred_xgb_train)
print("\n🚀 XGBoost - Prediksi IKP")
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, RMSPE={rmspe:.2f}%")

# Input manual prediksi tahun 2024–2025
input_manual_ikp = pd.DataFrame([
    {
        "Tahun": 2024,
        "PoU": 13.23,
        "Skor PPH": 91.17,
        "Konsumsi Energi (kkal/kap/hari)": 2026.07,
        "presentase_penduduk_kurang_gizi": 13.13
    },
    {
        "Tahun": 2025,
        "PoU": 13.64,
        "Skor PPH": 93.20,
        "Konsumsi Energi (kkal/kap/hari)": 2012.15,
        "presentase_penduduk_kurang_gizi": 13.55
    }
])

# Prediksi masa depan
pred_xgb_ikp = model_xgb.predict(input_manual_ikp)
input_manual_ikp["Prediksi_IKP"] = pred_xgb_ikp
print("\n📈 Prediksi IKP Tahun 2024–2025 (XGBoost):")
print(input_manual_ikp[["Tahun", "Prediksi_IKP"]])


from sklearn.model_selection import train_test_split

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Fit model pada data train
model_rf.fit(X_rf_train, y_rf_train)

# Prediksi di train dan test
y_train_pred = model_rf.predict(X_rf_train)
y_test_pred = model_rf.predict(X_rf_test)

# Evaluasi
print("🔵 TRAIN:")
print(evaluate_model(y_rf_train, y_train_pred))
print("🟡 TEST:")
print(evaluate_model(y_rf_test, y_test_pred))
