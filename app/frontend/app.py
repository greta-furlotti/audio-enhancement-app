import streamlit as st
import requests
import pandas as pd
from pathlib import Path

BACKEND = "http://localhost:8000"

st.title("Speech Enhancement App")

# ── Upload & denoise ──────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.subheader("Original audio")
    st.audio(uploaded_file)

    if st.button("Denoise"):
        with st.spinner("Processing..."):
            response = requests.post(
                f"{BACKEND}/denoise",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )

        if response.status_code == 200:
            data     = response.json()
            filename = data["output_filename"]

            audio_response = requests.get(f"{BACKEND}/audio/{filename}")

            if audio_response.status_code == 200:
                st.success("Done!")
                st.subheader("Enhanced audio")
                st.audio(audio_response.content, format="audio/wav")

                st.download_button(
                    label="Download enhanced WAV",
                    data=audio_response.content,
                    file_name=filename,
                    mime="audio/wav",
                )
            else:
                st.error(f"Could not retrieve audio: {audio_response.status_code}")
        else:
            st.error(f"Denoising failed ({response.status_code}): {response.text}")

# ── Metrics viewer ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Test set evaluation results")

METRICS_CSV = Path("data/processed/voicebank_demand/test/enhanced/metrics.csv")
SUMMARY_JSON = Path("data/processed/voicebank_demand/test/enhanced/summary.json")

if METRICS_CSV.exists():
    df = pd.read_csv(METRICS_CSV)

    # Summary row at the top
    if SUMMARY_JSON.exists():
        import json
        with open(SUMMARY_JSON) as f:
            summary = json.load(f)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files evaluated", summary["n_files"])
        col2.metric("Mean SNR",  f"{summary['snr_mean']} dB"  if summary['snr_mean']  else "n/a")
        col3.metric("Mean PESQ", str(summary['pesq_mean'])     if summary['pesq_mean'] else "n/a")
        col4.metric("Mean STOI", str(summary['stoi_mean'])     if summary['stoi_mean'] else "n/a")

    # Full table — sortable by clicking column headers in Streamlit
    st.dataframe(
        df,
        width='stretch',
        column_config={
            "file": st.column_config.TextColumn("File"),
            "snr":  st.column_config.NumberColumn("SNR (dB)",  format="%.3f"),
            "pesq": st.column_config.NumberColumn("PESQ",      format="%.3f"),
            "stoi": st.column_config.NumberColumn("STOI",      format="%.3f"),
        },
        hide_index=True,
    )

    # Download CSV directly from the UI
    with open(METRICS_CSV, "rb") as f:
        st.download_button(
            label="Download metrics.csv",
            data=f,
            file_name="metrics.csv",
            mime="text/csv",
        )
else:
    st.info("No evaluation results yet. Run `python -m src.evaluation.evaluate_testset` first.")