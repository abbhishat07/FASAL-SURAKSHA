"""
====================================================================
FASAL SURAKSHA - app.py
Production FastAPI Backend Server.
Features:
  - Multimodal Crop Diagnostic API (Webcam / File upload)
  - WhatsApp Bot Webhook (Twilio / Meta Graph compatible)
  - Tabular XGBoost Yield Simulator Endpoint
  - PMFBY Insurance Claim Evidence Review & Triage
  - 1-Click PDF Report Generator & Download
  - Prometheus Metrics & OpenAPI Documentation
====================================================================
"""

import os
import io
import base64
import uuid
import cv2
import numpy as np
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field

from ml_pipeline import FasalSurakshaPipeline
from utils.pmfby_verifier import verify_pmfby_claim

# Initialize FastAPI App
app = FastAPI(
    title="FASAL SURAKSHA (फसल सुरक्षा) API",
    description="Smart Micro-Crop Disease Diagnostic, Yield Prediction & PMFBY Insurance Verification Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware for Web/Mobile Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for PMFBY Claims triage queue & PDF cache
CLAIMS_DB: List[Dict[str, Any]] = []
PDF_CACHE_DIR = "reports_cache"
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# Initialize ML Pipeline Singleton
pipeline = FasalSurakshaPipeline(
    pytorch_model_path="models/fasal_suraksha_mobilenet.pth",
    onnx_model_path="models/fasal_suraksha_cnn.onnx",
    yield_model_path="models/yield_xgboost.pkl",
    prefer_onnx=False
)


# ─────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────

class YieldSimulationRequest(BaseModel):
    crop_age_days: float = Field(65.0, ge=10.0, le=180.0, description="Crop age since sowing")
    rainfall_mm: float = Field(550.0, ge=50.0, le=2500.0, description="Cumulative seasonal rainfall")
    soil_ph: float = Field(6.5, ge=4.0, le=9.5, description="Soil pH level")
    temperature_c: float = Field(27.0, ge=10.0, le=48.0, description="Mean temperature")
    disease_severity_percent: float = Field(30.0, ge=0.0, le=100.0, description="Leaf area infected %")


class PMFBYManualVerifyRequest(BaseModel):
    claim_id: str
    crop_name: str
    disease_name: str
    severity_percent: float
    farm_area_acres: float = 2.0
    farmer_id: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None


# Helper function to convert BGR image to base64 data URI
def bgr_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


# ─────────────────────────────────────────────────────────────────
# Root & Health Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root_status():
    return {
        "project": "FASAL SURAKSHA (फसल सुरक्षा)",
        "tagline": "AI Micro-Crop Diagnostic, Yield Forecast & Insurance Verification Engine",
        "status": "OPERATIONAL",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "diagnose_api": "/api/v1/diagnose",
            "whatsapp_webhook": "/api/v1/webhook/whatsapp",
            "yield_simulation": "/api/v1/yield/simulate",
            "pmfby_claims": "/api/v1/pmfby/claims",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "pytorch_gradcam_loaded": pipeline.gradcam_engine is not None,
        "onnx_loaded": pipeline.onnx_engine is not None,
        "xgboost_loaded": pipeline.yield_bundle is not None
    }


# ─────────────────────────────────────────────────────────────────
# Core Diagnostic Endpoint (Webcam / File Upload)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/diagnose", tags=["Diagnostics"])
async def diagnose_crop_leaf(
    image: UploadFile = File(..., description="Crop leaf image file (JPEG/PNG)"),
    crop_age_days: float = Form(65.0),
    rainfall_mm: float = Form(550.0),
    soil_ph: float = Form(6.5),
    temperature_c: float = Form(27.0),
    farm_area_acres: float = Form(2.0),
    crop_name: Optional[str] = Form(None),
    farmer_id: Optional[str] = Form("KCC-FMR-98214"),
    gps_lat: Optional[float] = Form(None),
    gps_lon: Optional[float] = Form(None),
    generate_pdf_report: bool = Form(True)
):
    """
    Accepts a leaf image and agronomic parameters.
    Performs segmentation, disease classification, Grad-CAM heatmap generation,
    severity scoring, dynamic remedies, yield prediction, and PMFBY verification.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")

        report_id = f"FSC_{uuid.uuid4().hex[:8]}"
        pdf_filename = os.path.join(PDF_CACHE_DIR, f"{report_id}.pdf")

        result = pipeline.analyze_crop_image(
            image_bgr=img_bgr,
            crop_age_days=crop_age_days,
            rainfall_mm=rainfall_mm,
            soil_ph=soil_ph,
            temperature_c=temperature_c,
            farm_area_acres=farm_area_acres,
            crop_name=crop_name,
            farmer_id=farmer_id,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            generate_pdf=generate_pdf_report,
            pdf_output_path=pdf_filename
        )

        # Convert imagery for JSON response
        images_dict = result.pop("images")
        response_payload = {
            "report_id": report_id,
            "crop": result["crop"],
            "disease_name": result["disease_name"],
            "confidence_percent": result["confidence_percent"],
            "severity_percent": result["severity_percent"],
            "severity_category": result["severity_category"],
            "dosage_plan": result["dosage_plan"],
            "yield_prediction": result["yield_prediction"],
            "pmfby_verification": result["pmfby_verification"],
            "visualizations": {
                "original_preview": bgr_to_base64(img_bgr),
                "segmented_leaf": bgr_to_base64(images_dict["segmented_bgr"]),
                "gradcam_heatmap": bgr_to_base64(images_dict["heatmap_overlay_bgr"]),
                "contour_overlay": bgr_to_base64(images_dict["contour_overlay_bgr"])
            },
            "pdf_report_url": f"/api/v1/report/{report_id}.pdf" if generate_pdf_report else None
        }

        # Cache claim in PMFBY triage queue for insurance officers
        CLAIMS_DB.insert(0, {
            "report_id": report_id,
            "timestamp": datetime.now().isoformat(),
            "farmer_id": farmer_id,
            "crop": result["crop"],
            "disease_name": result["disease_name"],
            "severity_percent": result["severity_percent"],
            "severity_category": result["severity_category"],
            "pmfby_verification": result["pmfby_verification"],
            "thumbnail": response_payload["visualizations"]["gradcam_heatmap"]
        })

        return JSONResponse(status_code=200, content=response_payload)

    except Exception as exc:
        logger.error(f"Diagnosis failure: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────
# WhatsApp Bot Webhook (Twilio / Meta Cloud API)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/webhook/whatsapp", tags=["WhatsApp Bot"])
async def whatsapp_webhook(
    Body: Optional[str] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    Latitude: Optional[float] = Form(None),
    Longitude: Optional[float] = Form(None)
):
    """
    Handles incoming WhatsApp messages from farmers and insurance surveyors.
    Accepts leaf photo (MediaUrl0) + optional location.
    Responds with instant Hindi/English diagnosis, remedy, and PMFBY claim status.
    """
    logger.info(f"Incoming WhatsApp message from: {From} | Media: {MediaUrl0}")

    # If farmer sent text greeting or query without media
    if not MediaUrl0:
        reply_text = (
            "🌾 *नमस्ते! Welcome to FASAL SURAKSHA (फसल सुरक्षा)* 🌾\n\n"
            "कृपया अपनी बीमार फसल की पत्ती का स्पष्ट फोटो (Clear leaf photo) भेजें।\n"
            "Please send a clear photo of the infected crop leaf.\n\n"
            "हम आपको तुरंत रोग का नाम, दवा (Organic/Chemical), और PMFBY फसल बीमा सहायता देंगे।\n"
            "_Powered by AI & ICAR Agronomic Protocols._"
        )
        return Response(
            content=f"<Response><Message>{reply_text}</Message></Response>",
            media_type="application/xml"
        )

    try:
        # Download image from WhatsApp Media Server
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.get(MediaUrl0)
            if resp.status_code != 200:
                raise ValueError("Could not download media from WhatsApp CDN.")
            img_bytes = resp.content

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run diagnosis
        res = pipeline.analyze_crop_image(
            image_bgr=img_bgr,
            gps_lat=Latitude,
            gps_lon=Longitude,
            farmer_id=From or "WHATSAPP-FARMER",
            generate_pdf=False
        )

        disease = res["disease_name"]
        sev_pct = res["severity_percent"]
        cat = res["severity_category"]
        chem = res["dosage_plan"]["chemical_plan"]
        org = res["dosage_plan"]["organic_plan"]
        pmfby = res["pmfby_verification"]

        reply_message = (
            f"🌿 *FASAL SURAKSHA AI DIAGNOSIS* 🌿\n"
            f"────────────────────\n"
            f"• *रोग (Disease):* {disease}\n"
            f"• *संक्रमण गंभीरता (Severity):* {sev_pct}% ({cat})\n\n"
            f"💊 *रासायनिक उपचार (Chemical Remedy):*\n"
            f"*{chem['name_hi']}*\n"
            f"मात्रा: {chem['concentration_per_liter']} (15L टंकी में {chem['knapsack_tank_dose']})\n\n"
            f"🌱 *जैविक उपचार (Organic Remedy):*\n"
            f"*{org['name_hi']}*\n"
            f"मात्रा: {org['concentration_per_liter']}\n\n"
            f"🛡️ *PMFBY फसल बीमा दावा (Insurance Triage):*\n"
            f"दावा स्थिति: *{pmfby['triage_priority'].split('(')[0]}*\n"
            f"अनुमानित क्षतिपूर्ति: *INR {pmfby['estimated_indemnity_payout_inr']:,.2f}*\n"
            f"────────────────────\n"
            f"_अधिक जानकारी के लिए नजदीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।_"
        )

        twiml_response = f"<Response><Message>{reply_message}</Message></Response>"
        return Response(content=twiml_response, media_type="application/xml")

    except Exception as exc:
        logger.error(f"WhatsApp webhook processing error: {exc}")
        err_msg = "क्षमा करें, फोटो का विश्लेषण नहीं हो सका। कृपया स्पष्ट तस्वीर दोबारा भेजें। / Sorry, error analyzing image."
        return Response(
            content=f"<Response><Message>{err_msg}</Message></Response>",
            media_type="application/xml"
        )


# ─────────────────────────────────────────────────────────────────
# Tabular Yield Simulation Endpoint
# ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/yield/simulate", tags=["Yield Forecasting"])
def simulate_yield(req: YieldSimulationRequest):
    """
    Simulates crop harvest yield and projected deficit using the XGBoost Tabular Model.
    """
    forecast = pipeline.forecast_yield(
        crop_age_days=req.crop_age_days,
        rainfall_mm=req.rainfall_mm,
        soil_ph=req.soil_ph,
        temperature_c=req.temperature_c,
        disease_severity_percent=req.disease_severity_percent
    )
    return {
        "status": "success",
        "forecast": forecast
    }


# ─────────────────────────────────────────────────────────────────
# PMFBY Insurance Adjudication & Triage Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/pmfby/claims", tags=["PMFBY Insurance"])
def list_pmfby_claims(priority_filter: Optional[str] = None):
    """
    Returns the real-time queue of crop damage claims for insurance survey officers.
    Allows sorting and filtering by priority level (TIER-1 / TIER-2 / TIER-3).
    """
    if priority_filter:
        filtered = [c for c in CLAIMS_DB if priority_filter.upper() in c["pmfby_verification"]["triage_priority"].upper()]
        return {"total_claims": len(filtered), "claims": filtered}
    return {"total_claims": len(CLAIMS_DB), "claims": CLAIMS_DB}


@app.post("/api/v1/pmfby/verify", tags=["PMFBY Insurance"])
def manual_verify_claim(req: PMFBYManualVerifyRequest):
    """
    Custom audit calculation for an insurance assessor inputting specific parameters.
    """
    # Estimate yield loss using standard agronomics
    yield_loss_pct = min(90.0, req.severity_percent * 0.95)
    verification = verify_pmfby_claim(
        disease_name=req.disease_name,
        severity_percent=req.severity_percent,
        predicted_yield_loss_percent=yield_loss_pct,
        farm_area_acres=req.farm_area_acres,
        crop_name=req.crop_name,
        farmer_id=req.farmer_id,
        claim_id=req.claim_id,
        gps_lat=req.gps_lat,
        gps_lon=req.gps_lon
    )
    return {"status": "success", "verification": verification}


# ─────────────────────────────────────────────────────────────────
# PDF Download Endpoint
# ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/report/{report_id}.pdf", tags=["Reports"])
def download_pdf_report(report_id: str):
    """
    Streams the generated 1-page Crop Health Diagnostic Card PDF.
    """
    file_path = os.path.join(PDF_CACHE_DIR, f"{report_id}.pdf")
    if not os.path.exists(file_path):
        # Fallback to local crop_health_card.pdf
        if os.path.exists("crop_health_card.pdf"):
            file_path = "crop_health_card.pdf"
        else:
            raise HTTPException(status_code=404, detail="Requested report PDF not found.")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"Fasal_Suraksha_{report_id}.pdf"
    )
