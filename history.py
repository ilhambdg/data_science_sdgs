from prophet import Prophet
import pandas as pd
import os
import glob
from functools import reduce
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

## -- baca file dari folder
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'dataset')
all_files = glob.glob(os.path.join(data_dir, '*.csv'))
smeua = [df1, df2, df3, df4, df5] = [pd.read_csv(f) for f in all_files]


## -- bersihin data
drop = ["No", "Wilayah", "Satuan", "Keterangan", "Kode Provinsi"]
ganti_nama = {
    'Kep Bangka Belitung': 'Kepulauan Bangka Belitung',
    'Kep Riau': 'Kepulauan Riau',
    'Ntb': 'Nusa Tenggara Barat',
    'Ntt': 'Nusa Tenggara Timur'
}


def kolom_bersih(df):
    for i in range(len(df)):    
        df[i] = df[i].drop(columns = [col for col in drop if col in df[i].columns], axis=1).dropna(subset='Provinsi')
        df[i]['Provinsi'] = df[i]['Provinsi'].apply(lambda x: x.title()).replace(ganti_nama)
        df[i] = df[i][df[i]['Provinsi'] != 'Kota Jambi']
    return df
df1, df2, df3, df4, df5 = kolom_bersih(smeua)


## -- gabungkan data 
df_predict = reduce(lambda left, right: pd.merge(left, right, on=['Tahun', "Provinsi"], how='outer'), kolom_bersih(smeua)).fillna(0)


df_predict['Jumlah_penduduk_gizi_cukup'] = df_predict['Jumlah_Penduduk'] - df_predict['Penduduk_Undernourish']
df_predict['persentase_penduduk_kurang_gizi'] = 100 - round((df_predict['Jumlah_penduduk_gizi_cukup'] / df_predict['Jumlah_Penduduk']) * 100)
df_predict['persentase_penduduk_gizi_cukup'] = round((df_predict['Jumlah_penduduk_gizi_cukup'] / df_predict['Jumlah_Penduduk']) * 100)
df_predict = df_predict[df_predict['Jumlah_Penduduk'] > df_predict['Penduduk_Undernourish']]


# # Data input (ganti sesuai kebutuhan)
# df_prov = df_predict[df_predict['Provinsi'] == 'Dki Jakarta'][['Tahun', 'PoU']].dropna()
# df_prov = df_prov.rename(columns={'Tahun': 'ds', 'PoU': 'y'})
# df_prov['ds'] = pd.to_datetime(df_prov['ds'].astype(str), format='%Y')

# # Latih Prophet
# model = Prophet()
# model.fit(df_prov)

# # Prediksi 5 tahun ke depan
# future = model.make_future_dataframe(periods=2, freq='Y')
# forecast = model.predict(future)

# # Plot hasil
# fig = model.plot(forecast)
# plt.title("Prediksi PoU oleh Prophet")
# plt.xlabel("Tahun")
# plt.ylabel("Prevalensi Undernourishment (PoU)")
# plt.grid(True)
# plt.show()



# Load dan siapkan data

df_predict['Tahun'] = pd.to_datetime(df_predict['Tahun'].astype(str), format='%Y')

prov = st.selectbox("Pilih Provinsi", df_predict['Provinsi'].unique())
dfp = df_predict[df_predict['Provinsi'] == prov][['Tahun', 'IKP']].rename(columns={'Tahun': 'ds', 'IKP': 'y'})

# Latih Prophet dan prediksi
model = Prophet()
model.fit(dfp)
future = model.make_future_dataframe(periods=3, freq='Y')
forecast = model.predict(future)

# Plot
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
ax.set_facecolor('#0e1117')
ax.scatter(dfp['ds'], dfp['y'], color='orange', s=60, label='Data Aktual')
ax.plot(forecast['ds'], forecast['yhat'], color='cyan', alpha=0.4, linewidth=2, label='Prediksi')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='cyan', alpha=0.1)
for spine in ax.spines.values(): spine.set_color('white')
ax.tick_params(colors='white')
ax.set_title(f"Prediksi IKP - {prov}", color='white')
ax.set_xlabel("Tahun", color='white')
ax.set_ylabel("PoU", color='white')
ax.legend()

# Ambil tahun-tahun aktual dari data latih
tahun_aktual = dfp['ds'].unique()

# Tampilkan label hanya untuk tahun future
for _, row in forecast.iterrows():
    if row['ds'] not in tahun_aktual:
        ax.text(row['ds'], row['yhat'] + 0.2, f"{row['yhat']:.2f}",
                fontsize=8, color='cyan', alpha=0.8, ha='center')

# Hitung evaluasi hanya untuk data aktual (tahun yang ada di dfp)
df_eval = pd.merge(
    dfp[['ds', 'y']],
    forecast[['ds', 'yhat']],
    on='ds',
    how='inner'
)

# Hitung MAE dan RMSE
mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
rmse = np.sqrt(mean_squared_error(df_eval['y'], df_eval['yhat']))

# Tambahkan teks evaluasi di grafik
ax.text(0.02, 0.95, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', edgecolor='white', alpha=0.6))

st.pyplot(fig)

