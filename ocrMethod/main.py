
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
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    return img, thresh
 
 
def encode_image_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return buffer.tobytes()
 
 