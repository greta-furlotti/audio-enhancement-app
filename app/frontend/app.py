import streamlit as st
import requests

st.title("Speech Enhancement App")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.subheader("Original Audio")
    st.audio(uploaded_file)

    if st.button("Denoise"):
        with st.spinner("Processing..."):
            response = requests.post(
                "http://localhost:8000/denoise",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )

        st.success("Done!")

        # not downloading the file right now: TODO
        st.write(response.json())