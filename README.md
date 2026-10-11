# 🌾 FASAL SURAKSHA (फसल सुरक्षा)
### Smart Micro-Crop Disease Diagnostic, Yield Prediction & PMFBY Insurance Verification Engine

![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-2e7d32.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-1b5e20.svg)
![Deep Learning](https://img.shields.io/badge/AI-PyTorch%20%7C%20Grad--CAM%20%7C%20ONNX%20%7C%20TFLite-43a047.svg)
![Insurance](https://img.shields.io/badge/PMFBY-Adjudication%20Norms%20Rev%202024-green.svg)

---

## 📖 1. Project Overview & UI Theme
**FASAL SURAKSHA (फसल सुरक्षा)** is an end-to-end, edge-ready agricultural diagnostics and insurance verification platform designed specifically for small-scale farmers, agronomists, and crop insurance survey officers (PMFBY).

- **Agricultural White & Green UI Theme:** Built with a high-contrast palette (`#1b5e20` deep emerald, `#2e7d32` vibrant leaf green, `#ffffff` card backdrops, `#e8f5e9` soft mint accents) optimized for high sunlight outdoor visibility in rural farm fields.
- **Multimodal Capture:** Accepts crop leaf images via **live webcam**, **drag-and-drop web upload**, or an incoming **WhatsApp Bot Webhook**.
- **Full Workflow:** Foliar contour segmentation $\rightarrow$ Disease classification $\rightarrow$ Grad-CAM lesion heatmaps $\rightarrow$ Disease Severity Index (DSI) $\rightarrow$ ICAR dynamic pesticide dosage $\rightarrow$ XGBoost harvest loss simulation $\rightarrow$ PMFBY insurance claim triage $\rightarrow$ 1-Click downloadable PDF report.

---

## 🏛️ 2. System Architecture

```text
                                 ┌───────────────────────────────┐
                                 │   FARMER / SURVEYOR INPUT     │
                                 │  (Webcam / Web / WhatsApp)    │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │      OPENCV SEGMENTATION      │
                                 │  • GrabCut & HSV Color Filter │
                                 │  • Leaf Isolation & DSI (%)   │
                                 └───────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐
         │     DISEASE CLASSIFIER       │                  │   EXPLAINABLE AI (XAI)       │
         │  • MobileNetV3-Small         │                  │  • Grad-CAM Conv Hooks       │
         │  • ONNX Runtime Graph        │                  │  • Glowing Heatmap Overlay   │
         │  • Quantized TFLite (<20MB)  │                  │  • Visual Trust Verification │
         └──────────────┬───────────────┘                  └──────────────┬───────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │   TABULAR ML YIELD ENGINE     │
                                 │  • XGBoost Regressor          │
                                 │  • Age, Rain, pH, Temp, DSI   │
                                 │  • Forecasts Actual Yield     │
                                 └───────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐
         │     SMART REMEDY ENGINE      │                  │   PMFBY INSURANCE VERIFIER   │
         │  • Dynamic Dosage Calculator │                  │  • Localized Loss Triage     │
         │  • Organic Bio-Solutions     │                  │  • Indemnity Estimation      │
         │  • Chemical Mix (mL/L & Tank)│                  │  • Priority 1/2/3 Queues     │
         └──────────────┬───────────────┘                  └──────────────┬───────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │   OUTPUT & DELIVERY CHANNELS  │
                                 │  • White/Green Streamlit App  │
                                 │  • Dual-Language Audio TTS    │
                                 │  • 1-Click PDF Card Export    │
                                 │  • WhatsApp Automated Reply   │
                                 └───────────────────────────────┘
```

---

## 📂 3. Project Directory Structure

```text
FASAL SURAKSHA/
├── dataset/                         # Training / Sample leaf pathology datasets
│   ├── Corn_(maize)___Common_rust_/
│   ├── Potato___Early_blight/
│   ├── Rice___Leaf_Blast/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
├── models/
│   ├── fasal_suraksha_cnn.onnx      # Cross-platform ONNX classifier
│   ├── fasal_suraksha_quant.tflite  # INT8/FP16 Quantized edge model (<20MB)
│   ├── fasal_suraksha_mobilenet.pth # PyTorch model weights with Grad-CAM hooks
│   └── yield_xgboost.pkl            # XGBoost Regressor bundle
├── utils/
│   ├── segmentation.py              # OpenCV background subtraction & DSI index
│   ├── gradcam.py                   # PyTorch Grad-CAM hook engine & overlay
│   ├── dosage_calculator.py         # ICAR remedy & dynamic dosage calculator
│   ├── pmfby_verifier.py            # PMFBY insurance claim triage & indemnity
│   └── pdf_generator.py             # 1-Page Crop Health Card PDF generator
├── train_cnn.py                     # MobileNetV3 training & ONNX export script
├── quantize.py                      # TFLite post-training quantization (<20MB)
├── train_yield.py                   # XGBoost yield regressor training script
├── ml_pipeline.py                   # Unified standardized inference pipeline
├── app.py                           # Production FastAPI backend & WhatsApp webhook
├── frontend.py                      # Streamlit dual-panel web dashboard
├── Dockerfile                       # Multi-stage production container configuration
└── requirements.txt                 # Complete pinned Python dependencies
```

---

## ⚡ 4. Core Features

### A. Dual-Panel Dashboard (`frontend.py`)
- **Left Panel (Foliar Input):**
  - High-resolution drag-and-drop file uploader
  - Live webcam capture module (`st.camera_input`)
  - 1-Click test pathology selector (Tomato Late Blight, Potato Early Blight, Corn Rust, Rice Blast, Healthy Leaf)
  - Real-time OpenCV segmentation preview displaying raw photo alongside isolated foliage contour
  - Agronomic inputs: Crop age, rainfall, soil pH, temperature, farm acreage, KCC farmer ID, and GPS coordinates
- **Right Panel (Diagnostics & Explainability):**
  - Side-by-side original leaf vs. **Grad-CAM glowing activation heatmap**
  - AI confidence meter (%) and Disease Severity Index (DSI) badge (Mild / Moderate / Severe)
  - **Smart Remedy Calculator:** Chemical and organic treatment instructions, dynamic mix ratios (mL/L), knapsack tank volume calculations, and safety pre-harvest intervals
  - **Dynamic Yield Risk Simulator:** Real-time XGBoost harvest forecast (Tons/Acre) with loss deficit meters
  - **PMFBY Insurance Evidence Dossier:** Evaluates localized calamity norms, flags claims above 20% threshold, estimates indemnity payout (INR), and assigns inspection priority (Tier-1 Urgent, Tier-2 Fast-Track, Tier-3 Sub-Threshold)
  - **Multi-Language Audio:** Hindi and English language toggles with Text-to-Speech (TTS) voice narration
  - **1-Click PDF Card:** Generates and downloads a clean 1-page "Crop Health Diagnostic Card" PDF

### B. Production FastAPI Backend (`app.py`)
- `POST /api/v1/diagnose`: Multimodal leaf analysis accepting image file and telemetry parameters. Returns complete JSON payload, base64 overlays, remedies, and PDF link.
- `POST /api/v1/webhook/whatsapp`: Twilio / Meta WhatsApp webhook endpoint for rural farmers. Receives leaf media, performs diagnosis, and replies with formatted Hindi/English advisories and insurance claim numbers.
- `POST /api/v1/yield/simulate`: Direct XGBoost agronomic tabular forecast endpoint.
- `GET /api/v1/pmfby/claims`: Real-time queue for insurance survey officers to audit claims prioritized by AI severity.
- `GET /api/v1/report/{report_id}.pdf`: Direct PDF download endpoint.

---

## 🚀 5. Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Bootstrap Models & Dataset
```bash
python bootstrap_models.py
```
*(Or train individual models via `python train_cnn.py`, `python quantize.py`, `python train_yield.py`)*

### 3. Run Web Dashboard (Streamlit)
```bash
streamlit run frontend.py --server.port 8501
```
Open browser at: `http://localhost:8501`

### 4. Run Backend API Server (FastAPI)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open interactive Swagger API documentation at: `http://localhost:8000/docs`

---

## 🐳 6. Docker Container Deployment

Build and run the unified production container (runs both FastAPI on 8000 and Streamlit on 8501):
```bash
docker build -t fasal-suraksha:latest .
docker run -p 8501:8501 -p 8000:8000 fasal-suraksha:latest
```

---

## 📜 7. Compliance & Standards
- **Agronomic Formulations:** Aligned with Indian Council of Agricultural Research (ICAR) & Directorate of Plant Protection, Quarantine & Storage (DPPQS) guidelines.
- **Crop Insurance Adjudication:** Formulated based on Pradhan Mantri Fasal Bima Yojana (PMFBY) Operational Guidelines for Localized Calamities and Post-Harvest Losses.
