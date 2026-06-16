import io
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = FastAPI()


def load_and_preprocess(image_bytes: bytes):
    """Load image bytes and return both the preprocessed (thresh) image and color original."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unable to load image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, thresh


def encode_image_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return buffer.tobytes()


@app.post("/call_ocr")
async def call_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, thresh = load_and_preprocess(image_bytes)

    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(thresh, lang="eng", config=config)

    return {"text": text.strip()}


@app.post("/preprocessed_preview")
async def preprocessed_preview(file: UploadFile = File(...)):
    """Return the preprocessed (thresholded) image that Tesseract actually sees."""
    image_bytes = await file.read()
    _, thresh = load_and_preprocess(image_bytes)

    img_bytes = encode_image_to_bytes(thresh)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.post("/boxed_image")
async def boxed_image(file: UploadFile = File(...)):
    """Return the original image with red bounding boxes around detected text regions."""
    image_bytes = await file.read()
    original, thresh = load_and_preprocess(image_bytes)

    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        thresh,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    # Scale factor: thresh was 2x upscaled; map boxes back to original size
    scale = 0.5

    # Draw a red box on the original for every word with confidence > 0
    annotated = original.copy()
    n = len(data["text"])

    for i in range(n):
        conf = int(data["conf"][i])
        word = data["text"][i].strip()

        if conf > 0 and word:
            x = int(data["left"][i] * scale)
            y = int(data["top"][i] * scale)
            w = int(data["width"][i] * scale)
            h = int(data["height"][i] * scale)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    img_bytes = encode_image_to_bytes(annotated)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")