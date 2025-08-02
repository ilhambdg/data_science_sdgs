import pandas as pd 
import os
import glob
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import reduce
import matplotlib.pyplot as plt
import streamlit as st
import plotly_express as px
import altair as alt
import plotly.graph_objs as go
import seaborn as sns

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

df_2023 = df_predict[df_predict['Tahun'] == 2023].sort_values(by='persentase_penduduk_gizi_cukup', ascending=False).to_numpy()

# === Fungsi evaluasi ===
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
   
def predict_with(df, year_col, value_col, label, groupby_avg=True):
    df = df[[year_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace(0, np.nan)
    if groupby_avg:
        df = df.groupby(year_col).mean().reset_index()
    df = df.dropna()
    df["ds"] = pd.to_datetime(df[year_col], format="%Y")
    df = df.rename(columns={value_col: "y"})

    model = Prophet(yearly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(df)

    forecast_train = model.predict(df[["ds"]])
    y_true = df["y"].values
    y_pred = forecast_train["yhat"].values
    mae, rmse, mape, rmspe = evaluate_model(y_true, y_pred)

    future = model.make_future_dataframe(periods=2026 - df["ds"].dt.year.max(), freq="Y")
    forecast = model.predict(future)

    output = forecast[["ds", "yhat"]].copy()
    output.columns = ["Tahun", f"Prediksi_{label}"]
    output["Tahun"] = output["Tahun"].dt.year
    output = output[output["Tahun"] > df[year_col].max()]
    return output

# === Fungsi Prophet + visualisasi ===
def predict_with_prophet(df, year_col, value_col, label, groupby_avg=True):
    df = df[[year_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace(0, np.nan)
    if groupby_avg:
        df = df.groupby(year_col).mean().reset_index()
    df = df.dropna()
    df["ds"] = pd.to_datetime(df[year_col], format="%Y")
    df = df.rename(columns={value_col: "y"})

    model = Prophet(yearly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(df)

    forecast_train = model.predict(df[["ds"]])
    y_true = df["y"].values
    y_pred = forecast_train["yhat"].values
    mae, rmse, mape, rmspe = evaluate_model(y_true, y_pred)

    future = model.make_future_dataframe(periods=2026 - df["ds"].dt.year.max(), freq="Y")
    forecast = model.predict(future)

    output = forecast[["ds", "yhat"]].copy()
    output.columns = ["Tahun", f"Prediksi_{label}"]
    output["Tahun"] = output["Tahun"].dt.year
    output = output[output["Tahun"] > df[year_col].max()]

 # === Visualisasi di Streamlit ===
      # Tampilkan evaluasi
    with st.expander(f"📊 Evaluasi Model - {label}"):
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**MAPE**: {mape:.2f}%")
        st.write(f"**RMSPE**: {rmspe:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode='markers+lines', name='Aktual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Prediksi', line=dict(color='orange', dash='dash')))

    fig.update_layout(
        title=f'📈 Prediksi vs Aktual - {label}',
        xaxis_title='Tahun',
        yaxis_title=value_col,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
  

    return output
 
def predict_with_Randomforest(df, target, fitur_col, df_predik, label, groupby_avg=True):
    if "Tahun" not in fitur_col:
        fitur_col = ["Tahun"] + fitur_col

    df = df[fitur_col + [target]].copy()
    df_predik = df_predik[fitur_col].copy()

    if groupby_avg:
        df = df.groupby("Tahun").mean().reset_index()

    model = RandomForestRegressor()
    model.fit(df[fitur_col], df[target])

    pred_train = model.predict(df[fitur_col])
    pred_future = model.predict(df_predik[fitur_col])

    y_true = df[target]
    mae, rmse, mape, rmspe = evaluate_model(y_true, pred_train)

    tahun_train = df["Tahun"]
    tahun_future = df_predik["Tahun"]

    # Gabungkan hasil
    tahun_all = pd.concat([tahun_train, tahun_future])
    pred_all = np.concatenate([pred_train, pred_future])
    aktual_all = pd.concat([y_true, pd.Series([np.nan]*len(pred_future))])

    # Visualisasi
    with st.expander(f"📊 Evaluasi Model - {label}"):
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**MAPE**: {mape:.2f}%")
        st.write(f"**RMSPE**: {rmspe:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tahun_all, y=aktual_all, name="Aktual", mode="markers+lines", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=tahun_all, y=pred_all, name="Prediksi", mode="lines+markers", line=dict(color="orange", dash="dash")))

    fig.update_layout(title=f"📈 Prediksi vs Aktual - {label}",
                      xaxis_title="Tahun", yaxis_title=target,
                      template="plotly_white", height=400)

    st.plotly_chart(fig, use_container_width=True)


## -- setup streamlit
st.set_page_config(
    page_title="Dashboard Interaktif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

## pilihan layar di samping
with st.container():
    st.sidebar.title("Data prediksi & analisis")
    halaman = st.sidebar.selectbox(
        "Halaman",
        ["Visual Data", "Korelasi Data", "Prediksi Data"]
    )

final_df = predict_with(df_predict, "Tahun", "IKP", "Indeks Ketahanan Pangan").merge(predict_with(df_predict, "Tahun", "PoU", "PoU"), on="Tahun", how="outer") \
                        .merge(predict_with(df_predict, "Tahun", "persentase_penduduk_kurang_gizi", "Kurang Gizi"), on="Tahun", how="outer") \
                        .merge(predict_with(df_predict, "Tahun", "Skor PPH", "PPH"), on="Tahun", how="outer") \
                        .merge(predict_with(df_predict, "Tahun", "Konsumsi Energi (kkal/kap/hari)", "Konsumsi Energi"), on="Tahun", how="outer") \
                        .merge(predict_with(df_predict, "Tahun", "Jumlah_Penduduk", "Jumlah Penduduk"), on="Tahun", how="outer")


## layar data visual
if halaman == "Visual Data":
    st.header('Visualisasi Data')
    st.write('---')
    pilihan_provinsi = st.multiselect(
    "Nama Provinsi",
    df_predict['Provinsi'].unique()
    )

    isi = df_predict[df_predict['Provinsi'].isin(pilihan_provinsi)]
    
    if isi.empty:
        st.write('tolong isi')
        st.dataframe(df_predict)
    else:
        st.write('---')
        st.subheader('Line Chart')
        col1, col2 = st.columns(2)
        with col1:
            # (line chart konsumsi per-kapita)
            fig = px.line(isi, x='Tahun', y='Konsumsi Energi (kkal/kap/hari)', color='Provinsi', title="Rata-rata konsumsi energi per kapita")
            st.plotly_chart(fig)
        with col2:
            fig3 = px.line(isi, x='Tahun', y='Ketersediaan Energi', color='Provinsi', title="Jumlah ketersediaan energi")
            st.plotly_chart(fig3)

        st.write('---')

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.line(isi, x='Tahun', y='Skor PPH', color='Provinsi', title='Pola pangan harapan')
            st.plotly_chart(fig2)
        with col2:
            fig5 = px.line(isi, x="Tahun", y="PoU", color='Provinsi', title='Presentase Prevalensi kekurangan gizi' )
            st.plotly_chart(fig5)

        st.write('---')

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.line(isi, x='Tahun', y='persentase_penduduk_kurang_gizi', color='Provinsi', title='Pola penduduk kurang gizi')
            st.plotly_chart(fig2)
        with col2:
            fig2 = px.line(isi, x='Tahun', y='persentase_penduduk_gizi_cukup', color='Provinsi', title='Pola penduduk gizi terpenuhi')
            st.plotly_chart(fig2)

        st.write('---')

        st.subheader('Bar Chart')
        col1, col2 = st.columns(2)
        with col1:
            # Ubah ke format long
            df_melted = isi.melt(
                id_vars=['Tahun'],
                value_vars=['Ketersediaan Energi Hewani', 'Ketersediaan Energi Nabati'],
                var_name='Jenis Energi',
                value_name='Energi(kkal)'
                )
            # Bikin plot
            fig4 = px.bar(
                df_melted,
                title= "Kategori dalam energi",
                x='Tahun',
                y='Energi(kkal)',
                color='Jenis Energi',
                barmode='stack'
                )
            st.plotly_chart(fig4)  
        with col2:
            melted = isi.melt(
                id_vars=["Tahun"],
                value_vars=['Penduduk_Undernourish', "Jumlah_penduduk_gizi_cukup"],
                var_name="Jenis Penduduk",
                value_name="Jumlah"
            )
            fig5 = px.bar(
                melted,
                title= 'Kategori penduduk kurang gizi & cukup',
                x="Tahun",
                y="Jumlah",
                color="Jenis Penduduk",
                barmode="stack"
            )
            st.plotly_chart(fig5)

        st.write('---')

## gunakan hexbin dan scatter plot untuk korelasi data(input hanya x dan y) mungkin sambil liat heatmap(input bisa beragam)
elif halaman == "Korelasi Data":
    st.header('Korelasi data')
    st.write('---')

    st.sidebar.write('---')
    st.sidebar.title('Plotly & Hexbin')
    pilih_tahun = st.sidebar.selectbox(
        "Pilih tahun",
        df_predict['Tahun'].unique()
    )
    pilih_X = st.sidebar.selectbox(
        "Pilihan X",
        df_predict.columns.drop(['Tahun', 'Provinsi']).unique()
    )
    pilih_Y = st.sidebar.selectbox(
        "Pilihan Y",
        df_predict.columns.drop(['Tahun', 'Provinsi']).unique()
    )

    sesuai = df_predict[df_predict['Tahun'] == pilih_tahun]

    col1, col2 = st.columns(2)
    with col1:
        st.container(border=True).subheader('Top Provinsi gizi tercukupi 2023:')
        st.container(border=True).write(f'{df_2023[0,1]}, {df_2023[0,15]}% \n\n {df_2023[1,1]}, {df_2023[1,15]}% \n\n {df_2023[2,1]}, {df_2023[2,15]}% \n\n {df_2023[3,1]}, {df_2023[3,15]}% \n\n ')
    with col2:
        st.container(border=True).subheader('top provinsi kurang gizi:')
        st.container(border=True).write(f'{df_2023[32,1]}, {df_2023[32,15]}% \n\n {df_2023[31,1]}, {df_2023[31,15]}% \n\n {df_2023[30,1]}, {df_2023[30,15]}% \n\n {df_2023[29,1]}, {df_2023[29,15]}% \n\n ')
    

    #plotly chart
    st.subheader('Plotly Chart')
    chart = alt.Chart(sesuai).mark_circle(size=60).encode(
    x= pilih_X,
    y= pilih_Y,
    color='Provinsi',
    tooltip=['Provinsi', pilih_X, pilih_Y]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.write('---')

    st.subheader('Hexbin Chart')
    ## hexbin chart
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    hb = ax.hexbin(sesuai[pilih_X].to_numpy(), sesuai[pilih_Y].to_numpy(), gridsize=25, cmap='inferno', edgecolors='black')

    # Tambahkan colorbar
    ax.set_facecolor('none')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Jumlah Titik', fontsize=6, color='white')

    ax.tick_params(colors='white', labelsize=5)
    plt.setp(cb.ax.get_yticklabels(), color='white', fontsize=7)

    ax.set_xlabel(pilih_X, fontsize=6, color='white')
    ax.set_ylabel(pilih_Y, fontsize=6, color='white')

    # Tampilkan di Streamlit
    st.pyplot(fig)

    st.write('---')
    st.write('')
    st.subheader('Heat Map')
    # Hitung korelasi
    df_numeric = df_predict.select_dtypes(include='number')
    df_predict = df_predict[['PoU', 'IKP', 'persentase_penduduk_kurang_gizi', 'Ketersediaan Energi', 'Jumlah_Penduduk', 'Konsumsi Energi (kkal/kap/hari)']]
    corr = df_predict.corr()

    # Gambar
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.style.use('dark_background')

    sns.heatmap(corr, annot=True, cmap="RdBu", fmt=".2f", ax=ax,
                linewidths=0.5, square=True,
                annot_kws={"color": "black", 'size':6},
                cbar_kws={"shrink": 0.7})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

    st.pyplot(fig)


# gunakan model untuk prediksi tren(hanya 1/2 fitur), kemudian buat model dengan input fitur prediksi tren tersebut.
else:
    st.header('Prediksi data by model')
    st.write('(model prediksi mengambil rata-rata provinsi)')
    st.write('---')
    st.sidebar.write('---')
    st.sidebar.title('Prophet & Random Forest')
    fitur = st.sidebar.selectbox('tipe input',['satu fitur', 'banyak fitur'])

    if fitur == 'satu fitur':
        st.container(border=True).write('Data yang di prediksi: \n\n IKP, PoU, persentase penduduk kurang gizi, Skor PPH, Konsumsi energi, jumlah penduduk')
        st.subheader('Prophet')
        predict_with_prophet(df_predict, "Tahun", "IKP", "Indeks Ketahanan Pangan")
        predict_with_prophet(df_predict, "Tahun", "PoU", "PoU")
        predict_with_prophet(df_predict, "Tahun", "persentase_penduduk_kurang_gizi", "Kurang Gizi")
        predict_with_prophet(df_predict, "Tahun", "Skor PPH", "PPH")
        predict_with_prophet(df_predict, "Tahun", "Konsumsi Energi (kkal/kap/hari)", "Konsumsi Energi")
        predict_with_prophet(df_predict, "Tahun", "Jumlah_Penduduk", "Jumlah Penduduk")
    
     
        st.subheader("📅 Tabel Prediksi Prophet sampai 2025")
        st.dataframe(final_df.round(2), use_container_width=True)
   
    else:
        st.subheader('Random Forest') 
    
        df_predik_rf = pd.DataFrame([
            {
                "Tahun": 2024,
                "PoU": 13.23,
                "Skor PPH": 91.17,
                "Konsumsi Energi (kkal/kap/hari)": 2026.07,
                "persentase_penduduk_kurang_gizi": 13.13
            }, 
            
            {
                "Tahun": 2025,
                "PoU": 13.64,
                "Skor PPH": 93.2,
                "Konsumsi Energi (kkal/kap/hari)": 2012.15,
                "persentase_penduduk_kurang_gizi": 13.55
            }
        ])
        df_predik_XG = pd.DataFrame([
        {
            'Tahun': 2024,
            'IKP': 77.45,
            'Skor PPH': 91.17,
            'Konsumsi Energi (kkal/kap/hari)': 2026.07,
            'Jumlah_Penduduk': 8592403.43

        }, 
        {
            'Tahun': 2025,
            'IKP': 78.93,
            'Skor PPH': 93.2,
            'Konsumsi Energi (kkal/kap/hari)': 2012.15,
            'Jumlah_Penduduk': 8736364.91

        }])
        fitur_rf = [
            "Tahun",
            "PoU",
            "Skor PPH",
            "Konsumsi Energi (kkal/kap/hari)",
            "persentase_penduduk_kurang_gizi"
        ]
        fitur_XG = [
            'Tahun',
            'IKP',
            'Skor PPH',
            'Konsumsi Energi (kkal/kap/hari)',
            'Jumlah_Penduduk'
        ]
        # data latih
        predict_with_Randomforest(df_predict, 'IKP', fitur_rf, df_predik_rf, 'IKP')
        
        predict_with_Randomforest(df_predict, 'PoU', fitur_XG, df_predik_XG, 'PoU')
        ##################################################


        # Gabungkan hasil
       
        st.subheader("📅 Tabel Prediksi Prophet sampai 2025")
        st.dataframe(final_df.round(2), use_container_width=True)
        st.write('Saya mengambil data dari prophet untuk di jadikan input Randomforest')
    
    
# Jalankan prediksi untuk tiap fitur
   


    

    
  
