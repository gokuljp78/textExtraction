import streamlit as st
import requests
import os
from PIL import Image

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="OCR Demo", layout="wide")
st.title("📄 OCR Text Extraction")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    if st.button("Extract Text"):
        file_bytes = uploaded_file.getvalue()
        file_tuple = (uploaded_file.name, file_bytes, uploaded_file.type)

        with st.spinner("Processing…"):
            # 1. Run OCR — get extracted text
            ocr_resp = requests.post(
                f"{API_BASE}/call_ocr",
                files={"file": file_tuple},
            )

            # 2. Get preprocessed (thresholded) image Tesseract used
            pre_resp = requests.post(
                f"{API_BASE}/preprocessed_preview",
                files={"file": file_tuple},
            )

            # 3. Get original image annotated with red bounding boxes
            box_resp = requests.post(
                f"{API_BASE}/boxed_image",
                files={"file": file_tuple},
            )

        # ── Layout — 3 images side by side ──────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🖼️ Original")
            st.image(Image.open(uploaded_file), use_container_width=True)

        with col2:
            st.subheader("🔲 Preprocessed")
            if pre_resp.status_code == 200:
                st.image(pre_resp.content, use_container_width=True)
            else:
                st.error(f"Preview error {pre_resp.status_code}: {pre_resp.text}")

        with col3:
            st.subheader("🟥 Boxed Regions")
            if box_resp.status_code == 200:
                st.image(box_resp.content, use_container_width=True)
            else:
                st.error(f"Boxed image error {box_resp.status_code}: {box_resp.text}")

        # ── Extracted text ───────────────────────────────────────
        st.subheader("📝 Extracted Text")
        if ocr_resp.status_code == 200:
            result = ocr_resp.json()
            st.text_area(
                label="OCR Output",
                value=result.get("text", ""),
                height=300,
            )
        else:
            st.error(f"OCR error {ocr_resp.status_code}: {ocr_resp.text}")