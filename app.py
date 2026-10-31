"""Modul 2B: Deployment LLM Groq untuk klasifikasi sentimen BRImo.

Pilihan prediksi CSV: 20, 50, 100, 200, atau seluruh baris.
Seluruh pilihan berjalan dengan satu tombol prediksi, tanpa tahapan batch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

from classifier import classify, load_config


HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Analisis Sentimen BRImo - Groq", page_icon="🤖")
st.title("🤖 Klasifikasi Sentimen Ulasan BRImo (Groq)")
st.caption("Modul 2B · AI model service · tidak menggunakan model DNN")

if not (HERE / "config.json").exists():
    st.error("config.json belum tersedia. Jalankan Modul 2A hingga lolos, kemudian salin config.json ke folder Modul 2B.")
    st.stop()

cfg = load_config(HERE / "config.json")
try:
    key = st.secrets["GROQ_API_KEY"]
except Exception:
    key = os.environ.get("GROQ_API_KEY")
if not key or "ISI_" in key:
    st.error("GROQ_API_KEY belum diatur di .streamlit/secrets.toml atau Secrets Streamlit Cloud.")
    st.stop()


@st.cache_resource
def client_from_key(api_key: str):
    return Groq(api_key=api_key)


client = client_from_key(key)
config_fingerprint = json.dumps(cfg, ensure_ascii=False, sort_keys=True)


@st.cache_data(show_spinner=False, ttl=3600)
def prediksi_cached(ulasan: str, fingerprint: str):
    # fingerprint memastikan cache prediksi tidak tertukar setelah config berubah.
    return classify(client, cfg, {"ulasan": ulasan})


@st.cache_data(show_spinner=False)
def contoh_csv():
    path = HERE / "brimo_googleplaystore_review.csv"
    raw = pd.read_csv(path, usecols=["content", "score"], low_memory=False)
    raw = raw.dropna(subset=["content"])
    raw["content"] = raw["content"].astype(str).str.strip()
    raw = raw[raw["content"].str.len() >= 8].drop_duplicates(subset=["content"])

    groups = []
    for ratings in ([1, 2], [4, 5]):
        group = raw[raw["score"].isin(ratings)]
        if not group.empty:
            groups.append(group.sample(n=min(3, len(group)), random_state=42))
    if not groups:
        raise ValueError("Tidak ditemukan ulasan yang cocok sebagai sampel.")
    sampel = pd.concat(groups, ignore_index=True).rename(columns={"content": "ulasan"})[["ulasan"]]
    return sampel.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.subheader("Konfigurasi evaluasi yang dikunci")
    st.code(cfg["model"])
    st.write("Mode:", cfg.get("metode", "pure_llm"))
    ev = cfg.get("evaluasi", {})
    if "f1_macro" in ev:
        st.metric("F1 Macro 200 data test", f"{ev['f1_macro']:.3f}")
    if "f1_dnn_pembanding" in ev:
        st.metric("F1 DNN 200 data yang sama", f"{ev['f1_dnn_pembanding']:.3f}")

st.download_button(
    "📥 Download contoh data uji CSV",
    data=contoh_csv(),
    file_name="contoh_ulasan_baru.csv",
    mime="text/csv",
)

t1, t2 = st.tabs(["Prediksi satu ulasan", "Prediksi CSV (pilih jumlah baris)"])

with t1:
    with st.form("form_one"):
        txt = st.text_area(
            "Masukkan ulasan BRImo",
            placeholder="Contoh: Aplikasinya sering error saat login",
            height=120,
        )
        pressed = st.form_submit_button("Prediksi dari Groq")
    if pressed:
        if not txt.strip():
            st.warning("Tuliskan ulasan terlebih dahulu.")
        else:
            try:
                with st.spinner("Menghubungi API Groq..."):
                    label = prediksi_cached(txt.strip(), config_fingerprint)
                if label == "Positif":
                    st.success(f"Prediksi: **{label}**")
                elif label == "Negatif":
                    st.error(f"Prediksi: **{label}**")
                else:
                    st.warning("Jawaban API tidak valid.")
            except Exception as exc:
                st.error(f"Permintaan API gagal: {type(exc).__name__}: {str(exc)[:200]}")

with t2:
    upload = st.file_uploader("Unggah CSV dengan kolom `ulasan` atau `content`", type=["csv"])
    if upload is not None:
        try:
            uploaded_bytes = upload.getvalue()
            df = pd.read_csv(io.BytesIO(uploaded_bytes), low_memory=False)
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")
            st.stop()

        df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        nama_kolom = {c.lower(): c for c in df.columns}
        if "ulasan" in nama_kolom:
            df = df.rename(columns={nama_kolom["ulasan"]: "ulasan"})
        elif "content" in nama_kolom:
            df = df.rename(columns={nama_kolom["content"]: "ulasan"})
        else:
            st.error("Kolom ulasan tidak ditemukan. Gunakan kolom `ulasan` atau `content`.")
            st.stop()

        jumlah_asli = len(df)
        pilihan = st.selectbox(
            "Jumlah baris yang ingin diprediksi",
            ["20 baris", "50 baris", "100 baris", "200 baris", "Semua baris"],
            index=0,
        )
        limit = {
            "20 baris": 20,
            "50 baris": 50,
            "100 baris": 100,
            "200 baris": 200,
            "Semua baris": None,
        }[pilihan]
        if limit is not None:
            df = df.head(limit).copy().reset_index(drop=True)
        else:
            df = df.copy().reset_index(drop=True)
        total = len(df)
        st.caption(f"Jumlah data pada file: {jumlah_asli:,} baris (pilihan: {total:,} baris).".replace(",", "."))

        if total == 0:
            st.warning("Tidak ada baris untuk diproses.")
            st.stop()

        # Pertahankan hasil ketika Streamlit rerun, misalnya saat mengunduh CSV.
        file_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        identitas = f"{file_hash}|{pilihan}|{config_fingerprint}"
        if st.session_state.get("brimo_csv_identitas") != identitas:
            st.session_state["brimo_csv_identitas"] = identitas
            st.session_state["brimo_csv_prediksi"] = []
        hasil = st.session_state["brimo_csv_prediksi"]
        selesai = len(hasil)

        if selesai:
            st.info(f"Sudah diprediksi: {selesai:,} dari {total:,} ulasan.".replace(",", "."))

        # Tidak ada proses batch. Tombol menjalankan seluruh pilihan yang dipilih.
        semua = limit is None
        izin = True
        if semua:
            st.warning(
                f"Mode Semua baris akan mengirim hingga {total:,} permintaan Groq "
                "(satu permintaan per ulasan unik yang belum tersimpan di cache). "
                "Ini dapat menghabiskan kuota API, memicu rate limit, dan proses dapat terputus. "
                "Prediksi dilakukan berurutan dengan satu klik, bukan satu permintaan API."
                .replace(",", ".")
            )
            izin = st.checkbox(
                "Saya memahami penggunaan API dan ingin memprediksi semua baris dalam satu proses."
            )
        elif total >= 100:
            st.info("Prediksi 100–200 baris memerlukan banyak panggilan API Groq dan dapat terkena rate limit.")

        if selesai < total:
            if st.button("Mulai prediksi CSV", type="primary", disabled=not izin):
                progress = st.progress(selesai / total)
                status = st.empty()
                error_terakhir = None

                # Satu klik menjalankan seluruh baris terpilih. Hasil disimpan
                # dalam session_state agar tetap tampil saat halaman rerun.
                for i in range(selesai, total):
                    ulasan = str(df.at[i, "ulasan"]) if pd.notna(df.at[i, "ulasan"]) else ""
                    ulasan = ulasan.strip()
                    if not ulasan:
                        pred = "invalid"
                    else:
                        try:
                            pred = prediksi_cached(ulasan, config_fingerprint)
                        except Exception as exc:
                            # Jangan terus mengirim permintaan ketika API sudah gagal.
                            error_terakhir = f"{type(exc).__name__}: {str(exc)[:240]}"
                            break
                    hasil.append(pred)
                    # Batasi frekuensi pembaruan tampilan agar lebih ringan pada
                    # dataset besar, tanpa membatasi jumlah data yang diproses.
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        progress.progress((i + 1) / total)
                        status.caption(
                            f"Memproses ulasan {i + 1:,} dari {total:,} ...".replace(",", ".")
                        )

                if error_terakhir:
                    st.error(
                        "Prediksi berhenti karena API bermasalah: " + error_terakhir
                        + ". Hasil yang sudah selesai tetap dapat diunduh. "
                        "Klik Mulai prediksi CSV lagi untuk melanjutkan dari data terakhir."
                    )
                elif len(hasil) == total:
                    st.success(f"Selesai. {total:,} ulasan diproses.".replace(",", "."))

        if hasil:
            result_df = df.iloc[:len(hasil)].copy()
            result_df["prediksi_sentimen"] = hasil
            st.subheader("Hasil prediksi CSV")
            st.caption("Pratinjau maksimal 100 baris. Unduhan berisi semua hasil yang telah diproses.")
            preview = result_df[['ulasan', 'prediksi_sentimen']].copy()
            preview.insert(0, 'No.', range(1, len(preview) + 1))
            st.dataframe(preview.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download hasil prediksi CSV",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_prediksi_groq.csv",
                mime="text/csv",
            )
            if len(hasil) < total:
                st.info(
                    f"Hasil sementara: {len(hasil):,}/{total:,} ulasan. "
                    "Kamu dapat melanjutkan pemrosesan dengan tombol Mulai prediksi CSV."
                    .replace(",", ".")
                )

            if st.button("Ulangi prediksi dari awal", type="secondary"):
                st.session_state["brimo_csv_prediksi"] = []
                st.rerun()

st.caption("API key tetap rahasia. Setiap perubahan konfigurasi memerlukan evaluasi ulang Modul 2A.")
