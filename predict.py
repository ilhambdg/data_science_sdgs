import pandas as pd 
import os
import glob
from functools import reduce
import streamlit as st
import plotly_express as px


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
        df[i] = df[i].drop(columns = [col for col in drop if col in df[i].columns], axis=1).dropna()
        df[i]['Provinsi'] = df[i]['Provinsi'].apply(lambda x: x.title()).replace(ganti_nama)
        df[i] = df[i][df[i]['Provinsi'] != 'Kota Jambi']
    return df
df1, df2, df3, df4, df5 = kolom_bersih(smeua)


## -- gabungkan data 
df_predict = reduce(lambda left, right: pd.merge(left, right, on=['Tahun', "Provinsi"], how='outer'), kolom_bersih(smeua))


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

## layar data visual

if halaman == "Visual Data":
    st.header('Visualisasi Data')
    st.write('---')
    pilihan_provinsi = st.multiselect(
    "Nama Provinsi",
    df_predict['Provinsi'].unique()
    )

    isi = df_predict[df_predict['Provinsi'].isin(pilihan_provinsi)]
    isi = isi[isi['Jumlah_Penduduk'] > isi['Penduduk_Undernourish']]
    isi['Penduduk gizi cukup'] = isi['Jumlah_Penduduk'] - isi['Penduduk_Undernourish']

    if isi.empty:
        st.write('tolong isi')
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
                value_vars=['Penduduk gizi cukup', "Penduduk_Undernourish"],
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

else:
    st.write('e')
    