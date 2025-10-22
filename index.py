import os
import re
import cv2
import time
import random
import numpy as np
from flask import Flask, request, render_template_string, jsonify, url_for
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from PIL import Image

# === OCR Engines ===
import pytesseract
import easyocr

# === Torch / GPU Detection ===
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
except Exception as e:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = f"Unavailable ({e})"

# === Firebase Setup ===
import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT_PATH = "parcel-pin-system-firebase-adminsdk-fbsvc-d3a1cd4a87.json"
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Flask App ===
app = Flask(__name__)
run_with_ngrok(app)  # auto start ngrok tunnel

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====== Global OCR Reader (initialized once) ======
EASYOCR_GPU = bool(CUDA_AVAILABLE)
print("==== Runtime Info ====")
print(f"Torch available: {TORCH_AVAILABLE}")
print(f"CUDA available : {CUDA_AVAILABLE}")
print(f"Device         : {CUDA_DEVICE_NAME}")
print(f"EasyOCR GPU    : {EASYOCR_GPU}")
print("======================")

EASYOCR_READER = easyocr.Reader(["en"], gpu=EASYOCR_GPU)

def warmup_easyocr():
    dummy = np.full((32, 128), 255, dtype=np.uint8)
    try:
        _ = EASYOCR_READER.readtext(dummy)
    except Exception:
        pass

warmup_easyocr()

# ========= Helper Functions =========
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(img)
            try:
                gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0)
                img_blur = gpu_blur.apply(gpu_mat)
            except Exception:
                img_blur = gpu_mat
            _, gpu_thresh = cv2.cuda.threshold(img_blur, 150, 255, cv2.THRESH_BINARY)
            thresh = gpu_thresh.download()
        else:
            raise RuntimeError("No CUDA device for OpenCV")
    except Exception:
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(img_blur, 150, 255, cv2.THRESH_BINARY)

    return thresh

SIX_DIGIT = re.compile(r"\b\d{6}\b")

def ocr_tesseract(image):
    text = pytesseract.image_to_string(image, config="--psm 6")
    return SIX_DIGIT.findall(text)

def ocr_easyocr(image):
    results = EASYOCR_READER.readtext(image)
    hits = []
    for _, text, _ in results:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 6 and digits.isdigit():
            hits.append(digits)
    return hits

def check_pin_in_db(pin):
    docs = db.collection("parcel_pins").where("pin", "==", pin).stream()
    for doc in docs:
        return doc.to_dict().get("phone")
    return None

def generate_unique_id():
    return str(random.randint(100000, 999999))

def save_parcel_record(phone):
    docs = db.collection("parcel_count").stream()
    count = sum(1 for _ in docs)

    parcel_no = count + 1
    unique_id = generate_unique_id()

    db.collection("parcel_count").add({
        "parcel_no": parcel_no,
        "unique_id": unique_id,
        "phone": phone,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    print(f"📦 Saved → No:{parcel_no}, ID:{unique_id}, Phone:{phone}")
    return parcel_no, unique_id

# ========= Flask Routes =========
@app.route("/")
def index():
    return render_template_string(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>📷 Parcel Scanner</title>
            <script src="https://unpkg.com/sweetalert/dist/sweetalert.min.js"></script>
            <style>
              body { text-align:center; font-family:sans-serif; }
              #log { margin-top:40px; text-align:left; width:90%; max-width:900px; margin:auto; 
                     font-size:16px; border:1px solid #aaa; padding:10px; border-radius:8px; 
                     background:#f9f9f9; height:220px; overflow-y:auto; }
              video { border:2px solid black; max-width:95vw; height:auto; }
            </style>
        </head>
        <body>
            <h2>📦 Parcel Scanner</h2>
            <video id="video" autoplay playsinline width="400" height="300"></video>
            <h3 id="result" style="margin-top:30px; margin-bottom:40px;">🔍 Waiting for scan...</h3>
            <div id="log"><b>📜 Scan Log:</b><br></div>
            <audio id="beep" src="{{ url_for('static', filename='scanner.mp3') }}" preload="auto"></audio>

            <script>
                const video = document.getElementById('video');
                const result = document.getElementById('result');
                const logDiv = document.getElementById('log');
                const beep = document.getElementById('beep');

                async function startBackCamera() {
                  try {
                    const constraints = { video: { facingMode: { ideal: "environment" } }, audio: false };
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = stream;
                  } catch (err) {
                    console.error("Error accessing back camera:", err);
                    swal("⚠ Camera Error", "Back camera not available. Please use a device with a rear camera.", "error");
                  }
                }

                function captureFrame() {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
                }

                async function sendFrame() {
                    try {
                        const blob = await captureFrame();
                        const formData = new FormData();
                        formData.append("frame", blob, "frame.jpg");

                        const res = await fetch("/scan", { method: "POST", body: formData });
                        const data = await res.json();

                        if (data.number && data.phone) {
                            result.innerHTML = `✅ PIN: ${data.number}<br>📱 Phone: ${data.phone}<br>📦 Parcel No: ${data.parcel_no}<br>🆔 ID: ${data.unique_id}`;
                            logDiv.innerHTML += `✔ Found PIN ${data.number} → Phone: ${data.phone}, Parcel: ${data.parcel_no}, ID: ${data.unique_id}<br>`;
                            try { beep.play().catch(()=>{}); } catch(e){}
                        } else if (data.number && !data.phone) {
                            result.innerHTML = `⚠ PIN detected but not found in DB: ${data.number}`;
                            logDiv.innerHTML += `⚠ PIN ${data.number} not in database<br>`;
                        } else {
                            result.innerText = "❌ No valid PIN detected.";
                        }
                        logDiv.scrollTop = logDiv.scrollHeight;
                    } catch (e) {
                        console.error(e);
                    }
                }

                startBackCamera();
                setInterval(sendFrame, 5000);
            </script>
        </body>
        </html>
        """
    )

@app.route("/scan", methods=["POST"])
def scan_frame():
    detected_number = None
    phone_number = None
    parcel_no = None
    unique_id = None

    if "frame" not in request.files:
        return jsonify({"number": None, "phone": None})

    file = request.files["frame"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    processed = preprocess_image(filepath)
    if processed is None:
        return jsonify({"number": None, "phone": None, "parcel_no": None, "unique_id": None})

    easy_matches = ocr_easyocr(processed)
    if easy_matches:
        detected_number = easy_matches[0]
    else:
        tess_matches = ocr_tesseract(processed)
        if tess_matches:
            detected_number = tess_matches[0]

    if detected_number:
        phone_number = check_pin_in_db(detected_number)
        if phone_number:
            parcel_no, unique_id = save_parcel_record(phone_number)
        else:
            print(f"⚠ PIN {detected_number} not in database")
    else:
        print("❌ No 6-digit number found")

    return jsonify({
        "number": detected_number,
        "phone": phone_number,
        "parcel_no": parcel_no,
        "unique_id": unique_id
    })

# ========= Run App =========
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print(f"[Boot] Torch version: {torch.__version__}")
        print(f"[Boot] CUDA available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"[Boot] CUDA device: {torch.cuda.get_device_name(0)}")
    app.run()
