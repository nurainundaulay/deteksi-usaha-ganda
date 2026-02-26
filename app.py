import streamlit as st
import pandas as pd
import numpy as np
import re
import networkx as nx
import io
from rapidfuzz import fuzz

# --- INISIALISASI MEMORI (SESSION STATE) ---
if 'hasil_df' not in st.session_state:
    st.session_state.hasil_df = None
if 'status_pencarian' not in st.session_state:
    st.session_state.status_pencarian = ""

# --- FUNGSI PEMBERSIHAN ---
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stopwords = ['pt', 'cv', 'ud', 'toko', 'warung', 'jasa', 'kios', 'bengkel']
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)

def clean_sumber(text):
    if pd.isna(text) or str(text).strip() == "": return "Kosong"
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --- KONFIGURASI HALAMAN UTAMA ---
st.set_page_config(page_title="Aplikasi Deteksi Usaha Ganda", layout="wide")
st.title("🔍 Aplikasi Deteksi Usaha Ganda (Fuzzy Matching)")
st.write("Unggah file direktori usaha, filter area, dan temukan kandidat usaha ganda.")

# --- 1. UPLOAD FILE ---
uploaded_file = st.file_uploader("📂 Upload file CSV Direktori Usaha", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, dtype={'idsbr': str})
        if 'gcs_result' in df.columns:
            df = df[df['gcs_result'].astype(str) != '4'].copy()
        return df

    df_utama = load_data(uploaded_file)
    st.success(f"File berhasil dimuat! Total data (tanpa status ganda): {len(df_utama)} baris.")

    # --- 2. FILTER KECAMATAN & DESA ---
    st.subheader("📍 Filter Wilayah")
    col1, col2 = st.columns(2)
    
    with col1:
        list_kec = sorted(df_utama['nmkec'].dropna().unique())
        # Fungsi callback untuk mereset hasil jika kecamatan diganti
        def reset_hasil():
            st.session_state.hasil_df = None
            st.session_state.status_pencarian = ""
            
        pilih_kec = st.selectbox("Pilih Kecamatan:", ["-- Pilih --"] + list_kec, on_change=reset_hasil)
    
    with col2:
        if pilih_kec != "-- Pilih --":
            df_kec = df_utama[df_utama['nmkec'] == pilih_kec]
            list_desa = sorted(df_kec['nmdesa'].dropna().unique())
            pilih_desa = st.selectbox("Pilih Desa:", ["-- Pilih --", "Semua Desa"] + list_desa, on_change=reset_hasil)
        else:
            pilih_desa = st.selectbox("Pilih Desa:", ["-- Pilih Kecamatan Dulu --"])

    # --- 3. EKSEKUSI PROSES ---
    if pilih_kec != "-- Pilih --" and pilih_desa not in ["-- Pilih --", "-- Pilih Kecamatan Dulu --"]:
        st.write("---")
        threshold = st.slider("⚙️ Batas Minimal Kemiripan (%)", min_value=70, max_value=100, value=85)
        
        if st.button("🚀 Mulai Deteksi Ganda"):
            with st.spinner("Sedang mencari kemiripan data... Mohon tunggu."):
                
                if pilih_desa == "Semua Desa":
                    df_filtered = df_utama[df_utama['nmkec'] == pilih_kec].copy()
                else:
                    df_filtered = df_utama[(df_utama['nmkec'] == pilih_kec) & (df_utama['nmdesa'] == pilih_desa)].copy()

                if len(df_filtered) < 2:
                    st.session_state.status_pencarian = "kurang_data"
                    st.session_state.hasil_df = None
                else:
                    df_filtered['nama_clean'] = df_filtered['nama_usaha'].apply(clean_text)
                    df_filtered['alamat_clean'] = df_filtered['alamat_usaha'].apply(clean_text)
                    df_filtered['sumber_data_clean'] = df_filtered['sumber_data'].apply(clean_sumber)

                    priority_map = {
                        "OSS - Badan Usaha": 1,
                        "OSS - Perorangan": 2,
                        "Dir Pajak": 3,
                        "Tambahan Daerah s.d 31 Desember 2025": 4
                    }
                    df_filtered['prioritas'] = df_filtered['sumber_data_clean'].map(lambda x: priority_map.get(x, 5))

                    records = df_filtered.to_dict('records')
                    edges = []
                    
                    for i in range(len(records)):
                        for j in range(i+1, len(records)):
                            rec1 = records[i]
                            rec2 = records[j]
                            
                            score_nama = fuzz.token_set_ratio(rec1['nama_clean'], rec2['nama_clean'])
                            score_alamat = fuzz.token_set_ratio(rec1['alamat_clean'], rec2['alamat_clean'])
                            final_score = (0.6 * score_nama) + (0.4 * score_alamat)
                            
                            if final_score >= threshold:
                                edges.append((rec1['idsbr'], rec2['idsbr'], final_score))

                    G = nx.Graph()
                    for u, v, w in edges:
                        G.add_edge(u, v, weight=w)
                    clusters = [c for c in nx.connected_components(G) if len(c) > 1]

                    output_rows = []
                    for idx, cluster in enumerate(clusters):
                        grup_id = f"Grup_{idx+1:03d}"
                        cluster_recs = df_filtered[df_filtered['idsbr'].isin(cluster)].sort_values(by=['prioritas'])
                        
                        main_nama = cluster_recs.iloc[0]['nama_clean']
                        main_alamat = cluster_recs.iloc[0]['alamat_clean']
                        
                        for i, row in enumerate(cluster_recs.itertuples()):
                            if i == 0:
                                skor_tampil = "100.00%"
                                rekomendasi = "Kandidat Utama"
                            else:
                                s_n = fuzz.token_set_ratio(row.nama_clean, main_nama)
                                s_a = fuzz.token_set_ratio(row.alamat_clean, main_alamat)
                                skor_angka = (0.6 * s_n) + (0.4 * s_a)
                                skor_tampil = f"{skor_angka:.2f}%"
                                rekomendasi = "Kandidat Ganda (4)"
                                
                            output_rows.append({
                                'grup_duplikat': grup_id,
                                'idsbr': row.idsbr,
                                'nama_usaha': row.nama_usaha,
                                'alamat_usaha': row.alamat_usaha,
                                'sumber_data': row.sumber_data,
                                'skor_kemiripan': skor_tampil,
                                'rekomendasi_sistem': rekomendasi,
                                'gcs_result': row.gcs_result,
                                'gc_username': row.gc_username
                            })

                    if len(output_rows) > 0:
                        st.session_state.hasil_df = pd.DataFrame(output_rows)
                        st.session_state.status_pencarian = "sukses"
                    else:
                        st.session_state.hasil_df = None
                        st.session_state.status_pencarian = "tidak_ada_ganda"

        # --- TAMPILKAN HASIL DARI MEMORI (Bukan dari dalam tombol) ---
        if st.session_state.status_pencarian == "kurang_data":
            st.warning("Data di wilayah ini kurang dari 2, tidak bisa dilakukan pencocokan.")
            
        elif st.session_state.status_pencarian == "tidak_ada_ganda":
            st.info("Tidak ada indikasi usaha ganda di wilayah ini berdasarkan persentase kemiripan tersebut.")
            
        elif st.session_state.status_pencarian == "sukses" and st.session_state.hasil_df is not None:
            df_output = st.session_state.hasil_df
            st.success(f"Selesai! Ditemukan {df_output['grup_duplikat'].nunique()} grup usaha ganda.")
            st.dataframe(df_output) 
            
            st.write("---")
            st.write("📥 **Unduh Hasil Deteksi:**")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv = df_output.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Unduh format CSV",
                    data=csv,
                    file_name=f'kandidat_ganda_{pilih_kec}_{pilih_desa}.csv',
                    mime='text/csv',
                )
                
            with col_dl2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_output.to_excel(writer, index=False, sheet_name='Kandidat Ganda')
                
                st.download_button(
                    label="📊 Unduh format Excel (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f'kandidat_ganda_{pilih_kec}_{pilih_desa}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )