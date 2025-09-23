import pandas as pd 
import os
import glob
import json
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

# -- datasetNasioanl
df_unicef = pd.read_csv("datasetNasional/fusion_NUTRITION_UNICEF_1.0_IDN.NT_ANT_HAZ_NE2_MOD.......csv")
df_poverty = pd.read_csv("datasetNasional/poverty_idn.csv")
df_asli = pd.read_csv("data_bersih.csv", sep=";")

# Filter kolom penting dari masing-masing dataset
unicef_filtered = df_unicef[[
    "TIME_PERIOD:Time period",
    "INDICATOR:Indicator",
    "OBS_VALUE:Observation Value",
    "REF_AREA:Geographic area"
]].rename(columns={"TIME_PERIOD:Time period" : "Year"})
unicef_filtered["Year"] = unicef_filtered["Year"].astype(int)
unicef_filtered = unicef_filtered.rename(columns={"Year" : "Tahun"})

# fao_filtered = df_fao[["Year", "Element", "Item", "Value"]]
# fao_filtered["Year"] = fao_filtered["Year"].astype(int)

poverty_filtered = df_poverty[["Year", "Indicator Name", "Value"]]
poverty_filtered = poverty_filtered.drop(poverty_filtered.index[poverty_filtered.index <= 11]).reset_index(drop=True)
poverty_filtered["Year"] = poverty_filtered["Year"].astype(int)
poverty_filtered = poverty_filtered.rename(columns={"Year" : "Tahun"})

# poverty_filtered.to_csv("data_poverty_clear.csv", index=False)

unicef_filtered = unicef_filtered.rename(columns={"OBS_VALUE:Observation Value" : "Value", "INDICATOR:Indicator" : "Indicator"})

unicef_filtered.loc[0:24, "Indicator"] = "(lower)Perkiraan anak yang mengalami stunting berdasar perbandingan usia dan tinggi badan"
unicef_filtered.loc[25:49, "Indicator"] = "(upper)Perkiraan anak yang mengalami stunting berdasar perbandingan usia dan tinggi badan"
unicef_filtered.loc[50:74, "Indicator"] = "(middle)Perkiraan anak yang mengalami stunting berdasar perbandingan usia dan tinggi badan"

# unicef_filtered.to_csv("data_stunting_clear.csv", index=False)

unicef_mid = unicef_filtered[unicef_filtered["Indicator"] == "(middle)Perkiraan anak yang mengalami stunting berdasar perbandingan usia dan tinggi badan"][["Tahun", "Value"]]
unicef_mid = unicef_mid.rename(columns={"Value" : "Perkiraan stunting by unicef"})

gini_df = poverty_filtered[poverty_filtered["Indicator Name"] == "Gini index"][["Tahun", "Value"]]
gini_df = gini_df.rename(columns={"Value": "Gini index"})

rasio_df = poverty_filtered[poverty_filtered["Indicator Name"] == "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"][["Tahun", "Value"]]
rasio_df = rasio_df.rename(columns={"Value" : "Rasio penduduk penghasilan < $3"})

df_predict = df_predict.merge(gini_df, on="Tahun", how="left").merge(unicef_mid, on="Tahun", how="left").merge(rasio_df, on="Tahun", how="left")
# metriksPoverty.loc[metriksPoverty["Year"] == tahun, "Value"]

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

    # forecast_train = model.predict(df[["ds"]])
    # y_true = df["y"].values
    # y_pred = forecast_train["yhat"].values
    # mae, rmse, mape, rmspe = evaluate_model(y_true, y_pred)

    future = model.make_future_dataframe(periods=2026 - df["ds"].dt.year.max(), freq="Y")
    forecast = model.predict(future)

    output = forecast[["ds", "yhat"]].copy()
    output.columns = ["Tahun", f"Prediksi_{label}"]
    output["Tahun"] = output["Tahun"].dt.year
    output = output[output["Tahun"] > df[year_col].max()]
    return output

def predict_with_2030(df, year_col, value_col, label, groupby_avg=True):
    df = df[[year_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace(0, np.nan)
    if groupby_avg:
        df = df.groupby(year_col).mean().reset_index()
    df = df.dropna()
    df["ds"] = pd.to_datetime(df[year_col], format="%Y")
    df = df.rename(columns={value_col: "y"})

    model = Prophet(yearly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.1)
    model.fit(df)

    future = model.make_future_dataframe(periods=2031 - df["ds"].dt.year.max(), freq="Y")
    forecast = model.predict(future)

    output = forecast[["ds", "yhat"]].copy()
    output.columns = ["Tahun", f"Prediksi_{label}"]
    output["Tahun"] = output["Tahun"].dt.year
    output = output[output["Tahun"] > df[year_col].max()]
    return output

# === Fungsi Prophet + visualisasi ===
def predict_with_prophet(df, year_col, value_col, label, badge, groupby_avg=True):
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

    with st.expander(f"📊 Evaluasi Model - {label}"):
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**MAPE**: {mape:.2f}%")
        st.write(f"**RMSPE**: {rmspe:.2f}%")

    if badge == 'baik':
        st.badge("Baik", icon=":material/check:", color="green")
    elif badge == 'stagnan':
        st.badge("Stagnan", icon=":material/warning:", color="gray")
    elif badge == 'buruk':
        st.badge("Buruk", icon=":material/error:", color="red")
    else: ''

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

def predict_with_prophet_2030(df, year_col, value_col, label, badge, groupby_avg=True): 
    df = df[[year_col, value_col]].copy() 
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce") 
    df = df.replace(0, np.nan) 
    if groupby_avg: 
        df = df.groupby(year_col).mean().reset_index() 
    df = df.dropna() 
    df["ds"] = pd.to_datetime(df[year_col], format="%Y") 
    df = df.rename(columns={value_col: "y"}) 

    model = Prophet(yearly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.1) 
    model.fit(df) 

    forecast_train = model.predict(df[["ds"]]) 
    y_true = df["y"].values 
    y_pred = forecast_train["yhat"].values 
    mae, rmse, mape, rmspe = evaluate_model(y_true, y_pred) 

    future = model.make_future_dataframe(periods=2031 - df["ds"].dt.year.max(), freq="Y") 
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

    if badge == 'baik': 
        st.badge("Baik", icon=":material/check:", color="green") 
    elif badge == 'stagnan': 
        st.badge("Stagnan", icon=":material/warning:", color="gray") 
    elif badge == 'buruk': 
        st.badge("Buruk", icon=":material/error:", color="red") 
    else: '' 

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

def predict_with_Randomforest(df, target, fitur_col, df_predik, label, badge, groupby_avg=True):
    if "Tahun" not in fitur_col:
        fitur_col = ["Tahun"] + fitur_col

    df = df[fitur_col + [target]].copy()
    df_predik = df_predik[fitur_col].copy()

    #menyamakan data aktual
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.replace(0, np.nan)
    if groupby_avg:
        df = df.groupby("Tahun").mean().reset_index()
    df = df.dropna()

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

    if badge == 'baik':
        st.badge("Baik", icon=":material/check:", color="green")
    elif badge == 'stagnan':
        st.badge("Stagnan", icon=":material/warning:", color="gray")
    else:
        st.badge("Buruk", icon=":material/error:", color="red")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tahun_train, y=aktual_all, name="Aktual", mode="markers+lines", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=tahun_all, y=pred_all, name="Prediksi", mode="lines", line=dict(color="orange", dash="dash")))

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

final_df_2030 = predict_with_2030(gini_df, "Tahun", "Gini index", "Gini index").merge(predict_with_2030(unicef_mid, "Tahun", "Perkiraan stunting by unicef", "Perkiraan stunting by unicef"), on="Tahun", how="outer") \
                        .merge(predict_with_2030(rasio_df, "Tahun", "Rasio penduduk penghasilan < $3", "Rasio penduduk penghasilan < $3"), on="Tahun", how="outer")
                       

## layar data visual
if halaman == "Visual Data":
    tab1, tab2 = st.tabs(["Data Pangan & gizi", "Data Kemiskinan & stunting anak"])
    with tab1:
        # st.header('Visualisasi Data')
        # st.write('---')

        col1, col2 = st.columns(2)
        with col1:
            st.container(border=True).subheader('Top provinsi gizi tercukupi (2023):')
            st.container(border=True).write(f'{df_2023[0,1]}, {df_2023[0,15]}% \n\n {df_2023[1,1]}, {df_2023[1,15]}% \n\n {df_2023[2,1]}, {df_2023[2,15]}% \n\n {df_2023[3,1]}, {df_2023[3,15]}% \n\n ')
        with col2:
            st.container(border=True).subheader('top provinsi kurang gizi (2023):')
            st.container(border=True).write(f'{df_2023[32,1]}, {df_2023[32,15]}% \n\n {df_2023[31,1]}, {df_2023[31,15]}% \n\n {df_2023[30,1]}, {df_2023[30,15]}% \n\n {df_2023[29,1]}, {df_2023[29,15]}% \n\n ')
        

        pilihan_provinsi = st.multiselect(
        "Nama Provinsi",
        df_predict['Provinsi'].unique()
        )

        mapping = {
        "DI. ACEH": "Aceh",
        "BALI": "Bali",
        "PROBANTEN": "Banten",
        "BENGKULU": "Bengkulu",
        "DAERAH ISTIMEWA YOGYAKARTA": "Di Yogyakarta",
        "DKI JAKARTA": "Dki Jakarta",
        "GORONTALO": "Gorontalo",
        "JAMBI": "Jambi",
        "JAWA BARAT": "Jawa Barat",
        "JAWA TENGAH": "Jawa Tengah",
        "JAWA TIMUR": "Jawa Timur",
        "KALIMANTAN BARAT": "Kalimantan Barat",
        "KALIMANTAN SELATAN": "Kalimantan Selatan",
        "KALIMANTAN TENGAH": "Kalimantan Tengah",
        "KALIMANTAN TIMUR": "Kalimantan Timur",
        "BANGKA BELITUNG": "Kepulauan Bangka Belitung",
        "LAMPUNG": "Lampung",
        "MALUKU": "Maluku",
        "MALUKU UTARA": "Maluku Utara",
        "NUSATENGGARA BARAT": "Nusa Tenggara Barat",
        "NUSA TENGGARA TIMUR": "Nusa Tenggara Timur",
        "RIAU": "Riau",
        "SULAWESI SELATAN": "Sulawesi Selatan",
        "SULAWESI TENGAH": "Sulawesi Tengah",
        "SULAWESI TENGGARA": "Sulawesi Tenggara",
        "SULAWESI UTARA": "Sulawesi Utara",
        "SUMATERA BARAT": "Sumatera Barat",
        "SUMATERA SELATAN": "Sumatera Selatan",
        "SUMATERA UTARA": "Sumatera Utara"
    }

        # 1. Baca file GeoJSON
        with open("indonesia-province-simple.json") as f:
            geojson = json.load(f)


        #testing cek nama provinsi apa saja
        for isi in geojson["features"]:
            a = isi["properties"]["Propinsi"]
            if a in mapping:
                isi["properties"]["Propinsi"] = mapping[a]
            
            
        prov_baru = ["IRIAN JAYA TIMUR", "IRIAN JAYA TENGAH", "IRIAN JAYA BARAT"]
        ikp_baru   = [0, 0, 0]  
        tahun = ["2023", "2023", "2023"]

        baris_baru = pd.DataFrame({
            "Provinsi": prov_baru,
            "Kelompok IKP": ikp_baru,
            "Tahun": tahun
        })

        df_asli = pd.concat([df_asli, baris_baru], ignore_index=True)



        isi = df_predict[df_predict['Provinsi'].isin(pilihan_provinsi)]
        
        if isi.empty:
            st.write('Silahkan Pilih Provinsi!')
          
        else:
            st.write('---')
            st.subheader('Tabel Pulau (nasional)')

             # 3. Visualisasi
            fig = px.choropleth(
                df_asli,
                title= "Indeks Ketahanan Pangan",
                geojson=geojson,
                locations="Provinsi",                 # kolom dari dataframe
                featureidkey="properties.Propinsi",   # sesuaikan field di GeoJSON
                color="Kelompok IKP",
                color_continuous_scale="inferno"
            )

            fig.update_geos(
                bgcolor="#333333",      # abu-abu gelap
                showland=True,
                landcolor="#333333"
            )

            # Atur tampilan peta
            fig.update_geos(fitbounds="locations", visible=False)

            # Tampilkan di Streamlit
            st.plotly_chart(fig, use_container_width=True)

            st.write("---")
            st.subheader('Line Chart (provinsi)')
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

            st.subheader('Bar Chart (provinsi)')
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
            st.write("Dataset.go.id")
            st.write('---')
    with tab2:
        poverty_filtered = poverty_filtered[poverty_filtered["Tahun"] >= 2000].reset_index(drop=True)
        poverty_filtered["Value"] = poverty_filtered["Value"].astype(float)
        poverty_filtered["valSebelum"] = poverty_filtered["Value"].shift(-1).astype(float)

        tahun = st.slider("Pilih tahun", 2001,2024)
        
        metriksPoverty = poverty_filtered[(poverty_filtered["Tahun"] == tahun) &(poverty_filtered["Indicator Name"] == "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)")]
        metriksPovertyGap = poverty_filtered[(poverty_filtered["Tahun"] == tahun) &(poverty_filtered["Indicator Name"] == "Poverty gap at $3.00 a day (2021 PPP) (%)")]
        metriksPovertyIndex = poverty_filtered[(poverty_filtered["Tahun"] == tahun) &(poverty_filtered["Indicator Name"] == "Gini index")]
    

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gini Index", metriksPovertyIndex.loc[metriksPovertyIndex["Tahun"] == tahun, "Value"].iloc[0], ((metriksPovertyIndex.loc[metriksPovertyIndex["Tahun"] == tahun, "Value"].iloc[0] - metriksPovertyIndex.loc[metriksPovertyIndex["Tahun"] == tahun, "valSebelum"].iloc[0]) / 100).round(3))
        with col2:
            st.metric("Rasio penduduk pengeluaran kurang dari 3$", metriksPoverty.loc[metriksPoverty["Tahun"] == tahun, "Value"].iloc[0], ((metriksPoverty.loc[metriksPoverty["Tahun"] == tahun, "Value"].iloc[0] - metriksPoverty.loc[metriksPoverty["Tahun"] == tahun, "valSebelum"].iloc[0]) / 100).round(3)) #berapa persen penduduk nasional 
        with col3:
            st.metric('poverty gap kurang dari $3 per hari', metriksPovertyGap.loc[metriksPovertyGap["Tahun"] == tahun, "Value"].iloc[0], ((metriksPovertyGap.loc[metriksPovertyGap["Tahun"] == tahun, "Value"].iloc[0] - metriksPovertyGap.loc[metriksPovertyGap["Tahun"] == tahun, "valSebelum"].iloc[0]) / 100).round(3)) #berapa rata2 pengeluaran/konsumsi yang di keluarkan yang kurang dari 3$
        # dari hal tersebut dapat terlihat apa mungkin melakukan solusi mbg dan juga pemberian sembako dapat efektif setidaknya mencapai titik
        # maksimum dari pemberian sembako agar prevalnsi stunting berkurang.

        povertySimpan = poverty_filtered[(poverty_filtered["Tahun"] == tahun) & (poverty_filtered["Indicator Name"].isin({
            "Income share held by second 20%",
            "Income share held by third 20%",
            "Income share held by fourth 20%",
            "Income share held by highest 20%",
            "Income share held by highest 10%",
            "Income share held by lowest 10%",
            "Income share held by lowest 20%"
        }))]

        fig = px.pie(povertySimpan, names="Indicator Name", values="Value", title="Pie Chart Total Kontribusi Value Pendapatan Nasional")
        st.plotly_chart(fig)
        st.write("World Bank - World Develpoment Indicators (WDI)")
        st.write("---")

        chart = (
        alt.Chart(unicef_filtered).mark_line(point=True).encode(
            
            x="Tahun:O",        # Tahun (ordinal)
            y="Value:Q",       # Nilai
            color="Indicator:N"   # Warna kategori
        )
        .properties(width=700, height=400, title="data stunting anak berdasar usia dan tinggi badan (model estimated by unicef)")
        )

        st.altair_chart(chart, use_container_width=True)
        st.write("UNICEF Data Portal")
        st.write("---")

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
        df_predict.columns.drop(['Tahun', 
                                 'Provinsi', 
                                 'Gini index', 
                                 "Perkiraan stunting by unicef", 
                                 "Rasio penduduk penghasilan < $3",
                                 "persentase_penduduk_kurang_gizi"]).unique()
    )
    pilih_Y = st.sidebar.selectbox(
        "Pilihan Y",
        df_predict.columns.drop(['Tahun', 
                                 'Provinsi', 
                                 "Gini index", 
                                 "Perkiraan stunting by unicef", 
                                 "Rasio penduduk penghasilan < $3",
                                 "persentase_penduduk_kurang_gizi"]).unique()
    )

    sesuai = df_predict[df_predict['Tahun'] == pilih_tahun]

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.container(border=True).subheader('Top provinsi gizi tercukupi (2023):')
    #     st.container(border=True).write(f'{df_2023[0,1]}, {df_2023[0,15]}% \n\n {df_2023[1,1]}, {df_2023[1,15]}% \n\n {df_2023[2,1]}, {df_2023[2,15]}% \n\n {df_2023[3,1]}, {df_2023[3,15]}% \n\n ')
    # with col2:
    #     st.container(border=True).subheader('top provinsi kurang gizi (2023):')
    #     st.container(border=True).write(f'{df_2023[32,1]}, {df_2023[32,15]}% \n\n {df_2023[31,1]}, {df_2023[31,15]}% \n\n {df_2023[30,1]}, {df_2023[30,15]}% \n\n {df_2023[29,1]}, {df_2023[29,15]}% \n\n ')
    

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

  
    ax.set_facecolor('none')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Jumlah Titik', fontsize=6, color='white')

    ax.tick_params(colors='white', labelsize=5)
    plt.setp(cb.ax.get_yticklabels(), color='white', fontsize=7)

    ax.set_xlabel(pilih_X, fontsize=6, color='white')
    ax.set_ylabel(pilih_Y, fontsize=6, color='white')

    st.pyplot(fig)


    st.write('---')
    st.write('')
    st.subheader('Heat Map')
    # df_numeric = df_predict.select_dtypes(include='number')
    df_predict = df_predict[['PoU', 
                             'IKP', 
                             'Ketersediaan Energi', 
                             'Jumlah_Penduduk', 
                             'Konsumsi Energi (kkal/kap/hari)', 
                             'Gini index', 
                             'Perkiraan stunting by unicef', 
                             "Rasio penduduk penghasilan < $3", 
                             "Tahun",
                             "Penduduk_Undernourish",
                             "Skor PPH"]]
   
    corr = df_predict.corr()

    # Gambar
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.style.use('dark_background')

    sns.heatmap(corr, annot=True, cmap="RdBu", fmt=".2f", ax=ax,
                linewidths=0.5, square=True,
                annot_kws={"color": "black", 'size':7},
                cbar_kws={"shrink": 0.7})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

    st.pyplot(fig)
    st.write("---")
    
    st.subheader("Fitur Korelasi")
    st.badge("IKP -> Stunting", icon=":material/check:", color="green")
    st.write("Data menunjukkan -0.74, yang berarti indeks ketahanan pangan (IKP) di suatu wilayah mempengaruhi tingkat stunting pada wilayah tersebut.")

    st.badge("Rasio penduduk penghasilan < 3$/hari -> IKP", icon=":material/check:", color="green")
    st.write("Data menunjukkan -0.73, yang berarti rasio pendapatan rendah akan mempengaruhi IKP pada suatu wilayah .")

    st.badge("Konsumsi energi harian -> PoU", icon=":material/check:", color="green")
    st.badge("Pertambahan penduduk kurang gizi -> pertambahan total penduduk", icon=":material/check:", color="green")
    st.write("Data menunjukkan 0.86, yang berarti setiap pertambahan jumlah penduduk di suatu daerah maka besar kemungkinan akan menambah jumlah penduduk kurang gizi, salah satu cara adalah bagaimana meningkatkan konsumsi energi harian penduduk " \
    "sesuai dengan korelasi PoU dengan konsumsi energi")


    st.badge("PoU != Stunting", icon=":material/error:", color="red")
    st.write("Ternyata persentase populasi kekurangan gizi (PoU) tidak dapat di tarik kesimpulan karena rendah nya korelasi, jika stunting pasti kekurangan gizi namun jika kekurangan gizi belum tentu stunting"
    "jadi masih ada harapan untuk memperbaiki fase stunting terutama usia di bawah 10 tahun.")
    st.badge("Gini index & Tahun", icon=":material/error:", color="red")
    st.write("Gini index menunjukkan besarnya ketimpangan masyrakat dimana berkorelasi dengan IKP dan sampai akhirnya dapat terjadi stunting. Ini " \
    "dapat menjadi poin dalam menurunkan tingkat stunting.")


    st.badge("penghasilan < $3 -> rasio stunting anak", color="blue")
    st.badge("Tahun -> rasio stunting anak & penghasilan < $3", color="blue")
    st.write("Terlihat setiap pertambahan tahun sangat berkorelasi dengan stunting dan juga penghasilan rendah, sebesar -0.99 yang menandakan bahwa kebijakan sudah sangat bagus dalam menurunkan tingkat stunting " \
    "dan juga terjadi penurunan masyarakat berpenghasilan rendah.")



# gunakan model untuk prediksi tren(hanya 1/2 fitur), kemudian buat model dengan input fitur prediksi tren tersebut.
else:
    st.header('Prediksi data by model')
    st.write('(model prediksi mengambil rata-rata provinsi)')
    col1, col2, col3 = st.columns(3)
    with col1 :
        st.badge("Baik", icon=":material/check:", color="green")
        st.write('Tren positif bisa dipertahankan dan dapat menjadi acuan solusi!')
    with col2:
        st.badge("Stagnan", icon=":material/warning:", color="gray")
        st.write('Tren stagnan memerlukan intervensi untuk peningkatan lebih baik!')
    with col3:
        st.badge("Buruk", icon=":material/error:", color="red")
        st.write('Tren negatif memerlukan intervensi secepatnya dalam kebijakan solusi!')

    st.write('---')
    st.sidebar.write('---')
    st.sidebar.title('Prophet & Random Forest')
    fitur = st.sidebar.selectbox('tipe input',['satu fitur', 'banyak fitur'])

    if fitur == 'satu fitur':
        st.container(border=True).write('Data yang di prediksi: \n\n IKP, PoU, persentase penduduk kurang gizi, Skor PPH, Konsumsi energi, jumlah penduduk')
        st.subheader('Prophet')
        predict_with_prophet(df_predict, "Tahun", "IKP", 'IKP', 'baik')
        predict_with_prophet(df_predict, "Tahun", "PoU", "PoU", 'buruk')
        predict_with_prophet(df_predict, "Tahun", "persentase_penduduk_kurang_gizi", "Kurang Gizi", 'buruk')
        predict_with_prophet(df_predict, "Tahun", "Skor PPH", "PPH", 'baik')
        predict_with_prophet(df_predict, "Tahun", "Konsumsi Energi (kkal/kap/hari)", "Konsumsi Energi", 'buruk')
        predict_with_prophet(df_predict, "Tahun", "Jumlah_Penduduk", "Jumlah Penduduk", 'tidak tau')
     
        st.subheader("📅 Tabel Prediksi Prophet sampai 2025")
        st.dataframe(final_df.round(2), use_container_width=True)
        st.write("---")

        predict_with_prophet_2030(gini_df, "Tahun", "Gini index", "Gini index", "buruk")
        predict_with_prophet_2030(unicef_mid, "Tahun", "Perkiraan stunting by unicef", "Perkiraan stunting by unicef", "baik")
        predict_with_prophet_2030(rasio_df, "Tahun", "Rasio penduduk penghasilan < $3", "Rasio penduduk penghasilan < $3", "baik")

        st.subheader("📅 Tabel Prediksi Prophet sampai 2030")
        st.dataframe(final_df_2030.round(2), use_container_width=True)
        st.write("---")
   
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
        predict_with_Randomforest(df_predict, "IKP", fitur_rf, df_predik_rf, "IKP", 'baik')
   
        
        predict_with_Randomforest(df_predict, 'PoU', fitur_XG, df_predik_XG, 'PoU', 'buruk')
        
        ##################################################


        # Gabungkan hasil
       
        st.subheader("📅 Tabel Prediksi Prophet sampai 2025")
        st.dataframe(final_df.round(2), use_container_width=True)
        st.write('Saya mengambil data dari prophet untuk di jadikan input Randomforest')
    
    

    
  
