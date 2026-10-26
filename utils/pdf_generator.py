"""
====================================================================
FASAL SURAKSHA - utils/pdf_generator.py   (revised)
One-page "Crop Health Diagnostic Card" PDF.

What changed vs. the previous version
-------------------------------------
* No made-up fallback values.  Missing data prints "N/A" (the old code
  showed Late Blight / 94.2 % / INR 22,425 when data was absent).
* Images go through memory buffers, not temp files in the working
  directory (safe for many simultaneous users).
* Insurance section is now an EVIDENCE RECORD: no payout, no eligibility,
  no "Geo: Verified" / "Tamper Score" unless real data exists.
* Wording fixed: Grad-CAM is "model attention", not verification; the
  footer no longer claims PMFBY / ICAR compliance.
* Hindi: the built-in Helvetica cannot draw Devanagari (you get boxes).
  Hindi text is OFF by default.  Set include_hindi=True and put a
  Devanagari TTF (e.g. NotoSansDevanagari-Regular.ttf) in assets/fonts/.
  NOTE: reportlab does not do complex-script shaping, so check the
  rendered Hindi carefully; if it looks wrong keep Hindi in the app UI/TTS.
====================================================================
"""
import io
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
from loguru import logger
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PRIMARY_GREEN = colors.HexColor("#1b5e20")
ACCENT_GREEN = colors.HexColor("#2e7d32")
LIGHT_BG_GREEN = colors.HexColor("#e8f5e9")
DARK_TEXT = colors.HexColor("#1a1a1a")
MUTED_TEXT = colors.HexColor("#424242")
BORDER_GREEN = colors.HexColor("#81c784")

_HINDI_FONT_NAME = "NotoDevanagari"
_HINDI_FONT_PATHS = [
    os.path.join("assets", "fonts", "NotoSansDevanagari-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
]


def _register_hindi_font() -> bool:
    if _HINDI_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return True
    for p in _HINDI_FONT_PATHS:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont(_HINDI_FONT_NAME, p))
                return True
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Could not register Hindi font {p}: {exc}")
    return False


def _fmt(value: Any, suffix: str = "", digits: Optional[int] = None) -> str:
    """Format a value, or 'N/A' when it is missing."""
    if value is None or value == "":
        return "N/A"
    if digits is not None:
        try:
            return f"{float(value):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"
    return f"{value}{suffix}"


def _esc(text: Any) -> str:
    """Escape text for reportlab's mini-HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bgr_to_flowable(image_bgr: Optional[np.ndarray], width: float, height: float, placeholder: Paragraph):
    if image_bgr is None:
        return placeholder
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return placeholder
    return Image(io.BytesIO(buf.tobytes()), width=width, height=height)


def generate_crop_health_pdf(
    diagnostic_data: Dict[str, Any],
    original_img_bgr: Optional[np.ndarray] = None,
    heatmap_img_bgr: Optional[np.ndarray] = None,
    output_filename: Optional[str] = "crop_health_card.pdf",
    include_hindi: bool = False,
) -> str:
    """Build the 1-page card and return the absolute PDF path."""
    if not output_filename:
        fd, output_filename = tempfile.mkstemp(suffix=".pdf", prefix="fasal_card_")
        os.close(fd)
    pdf_path = os.path.abspath(output_filename)

    use_hindi = include_hindi and _register_hindi_font()
    if include_hindi and not use_hindi:
        logger.warning("include_hindi=True but no Devanagari font found; Hindi text omitted.")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=26, rightMargin=26, topMargin=22, bottomMargin=22)
    ss = getSampleStyleSheet()

    title_style = ParagraphStyle("DocTitle", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                                 leading=18, textColor=PRIMARY_GREEN, spaceAfter=2)
    sub_style = ParagraphStyle("DocSub", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5, leading=11,
                               textColor=MUTED_TEXT, spaceAfter=6)
    section = ParagraphStyle("Section", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=13,
                             textColor=ACCENT_GREEN, spaceBefore=4, spaceAfter=3)
    body = ParagraphStyle("Body", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=10.5,
                          textColor=DARK_TEXT)
    bold = ParagraphStyle("Bold", parent=body, fontName="Helvetica-Bold")
    hindi_style = ParagraphStyle("Hindi", parent=body, fontName=_HINDI_FONT_NAME if use_hindi else "Helvetica")

    d = diagnostic_data or {}
    els = []

    # 1. header ---------------------------------------------------------
    now = datetime.now()
    hindi_line = (f"<br/><font name='{_HINDI_FONT_NAME}' size=11>फसल सुरक्षा</font>" if use_hindi else "")
    header = Table([[
        Paragraph(f"<b>FASAL SURAKSHA</b>{hindi_line}<br/><font size=8 color='#2e7d32'>"
                  "AI Crop Health Screening Card</font>", title_style),
        Paragraph(f"<b>Issued:</b> {now.strftime('%d %b %Y, %H:%M')}<br/>"
                  f"<b>Card ID:</b> FSC-{int(now.timestamp()) % 1000000:06d}<br/>"
                  "<b>Type:</b> AI screening - not a lab diagnosis", sub_style),
    ]], colWidths=[380, 180])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    els += [header, HRFlowable(width="100%", thickness=1.5, color=PRIMARY_GREEN, spaceBefore=2, spaceAfter=6)]

    # 2. summary bar ----------------------------------------------------
    sev_val = d.get("severity_percent")
    sev_txt = (f"{_fmt(sev_val, '%', 1)} ({_esc(d.get('severity_category', 'N/A'))})"
               if sev_val is not None else "N/A")
    meta = Table([[
        Paragraph("<b>Crop:</b>", bold), Paragraph(_esc(_fmt(d.get("crop"))), body),
        Paragraph("<b>Likely disease:</b>", bold), Paragraph(_esc(_fmt(d.get("disease_name"))), body),
        Paragraph("<b>Leaf severity:</b>", bold), Paragraph(f"<b>{sev_txt}</b>", body),
        Paragraph("<b>AI confidence:</b>", bold), Paragraph(_fmt(d.get("confidence_percent"), "%", 1), body),
    ]], colWidths=[40, 65, 75, 105, 70, 85, 75, 45])
    meta.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG_GREEN),
                              ("BOX", (0, 0), (-1, -1), 1, BORDER_GREEN),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    els += [meta, Spacer(1, 4)]

    low_conf_note = d.get("reliability_note")
    if low_conf_note and low_conf_note != "OK":
        els.append(Paragraph(f"<font color='#c62828'><b>Caution:</b> {_esc(low_conf_note)}</font>", body))
        els.append(Spacer(1, 4))

    # 3. images + yield -------------------------------------------------
    y = d.get("yield_prediction") or {}
    if y.get("predicted_yield_tons_per_acre") is not None:
        yield_par = Paragraph(
            f"<b>Reference yield (healthy):</b> {_fmt(y.get('baseline_potential_yield_tons'))} t/acre<br/>"
            f"<b>Model estimate:</b> {_fmt(y.get('predicted_yield_tons_per_acre'))} t/acre<br/>"
            f"<b>Estimated shortfall:</b> {_fmt(y.get('projected_loss_percent'), '%')}<br/><br/>"
            "<i>Statistical estimate from the inputs entered; it is not a measured loss and varies widely "
            "with weather, variety and management.</i>", body)
    else:
        yield_par = Paragraph("<i>No yield estimate available for this scan.</i>", body)

    img_l = _bgr_to_flowable(original_img_bgr, 1.85 * inch, 1.5 * inch, Paragraph("[Leaf photo]", body))
    img_r = _bgr_to_flowable(heatmap_img_bgr, 1.85 * inch, 1.5 * inch, Paragraph("[Heatmap]", body))
    vis = Table([
        [Paragraph("<b>Leaf photo</b>", section),
         Paragraph("<b>Model attention (Grad-CAM)</b>", section),
         Paragraph("<b>Yield estimate</b>", section)],
        [img_l, img_r, yield_par],
    ], colWidths=[150, 150, 260])
    vis.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 1, BORDER_GREEN)]))
    els += [vis, Paragraph("<i>Heatmap shows where the AI looked. It helps a human check the result; "
                           "it does not prove the diagnosis is correct.</i>", sub_style)]

    # 4. treatment --------------------------------------------------------
    els.append(Paragraph("<b>Suggested Treatment (advisory - confirm locally)</b>", section))
    dosage = d.get("dosage_plan") or {}
    org, chem = dosage.get("organic_plan") or {}, dosage.get("chemical_plan") or {}

    def plan_row(label, color, p, safety=""):
        hi = f"<br/><font name='{_HINDI_FONT_NAME}'>{_esc(p.get('name_hi', ''))}</font>" if (use_hindi and p.get("name_hi")) else ""
        return [
            Paragraph(f"<font color='{color}'><b>{label}</b></font>", body),
            Paragraph(f"<b>{_esc(_fmt(p.get('name_en')))}</b>{hi}", body),
            Paragraph(f"- {_esc(_fmt(p.get('concentration_per_liter')))}<br/>- {_esc(_fmt(p.get('knapsack_tank_dose')))}", body),
            Paragraph(f"Every {_esc(_fmt(p.get('interval_days')))} days.<br/>{_esc(safety or p.get('instructions_en', ''))}", body),
        ]

    rem = Table([
        [Paragraph("<b>Type</b>", bold), Paragraph("<b>Product</b>", bold),
         Paragraph("<b>Dose</b>", bold), Paragraph("<b>Frequency / safety</b>", bold)],
        plan_row("Organic / Bio", "#2e7d32", org),
        plan_row("Chemical", "#c62828", chem, d.get("safety_instructions_en", "")),
    ], colWidths=[85, 175, 145, 155])
    rem.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG_GREEN), ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREEN),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    els += [rem, Paragraph("<i>Use only pesticides registered for this crop in India, follow the product label, "
                           "and confirm the plan with a local agronomist or Krishi Vigyan Kendra (KVK).</i>", sub_style)]

    # 5. evidence record (no payout, no eligibility) -------------------------
    els.append(Paragraph("<b>Evidence Record (for crop-insurance reporting)</b>", section))
    ev = d.get("pmfby_verification") or d.get("evidence_record") or {}
    meta_ev = ev.get("audit_metadata") or {}
    missing = ev.get("evidence_missing") or []
    geo = meta_ev.get("geo_coordinates")
    ev_rows = [[
        Paragraph(f"<b>Record ID:</b><br/>{_esc(_fmt(ev.get('claim_id')))}", body),
        Paragraph(f"<b>Farmer / insurance ID:</b><br/>{_esc(_fmt(ev.get('farmer_id')))}", body),
        Paragraph(f"<b>Location:</b><br/>{_esc(geo) if geo else 'Not provided'}", body),
        Paragraph(f"<b>Evidence status:</b><br/>{_esc(_fmt(ev.get('evidence_status')))}", bold),
    ], [
        Paragraph(f"<b>Follow-up priority:</b><br/>{_esc(_fmt(ev.get('follow_up_priority')))}", body),
        Paragraph("<b>Still needed:</b><br/>" + (_esc("; ".join(missing)) if missing else "Nothing flagged"), body),
        Paragraph(f"<b>What to do:</b><br/>{_esc(_fmt(ev.get('surveyor_recommendation')))}", body),
        Paragraph(f"<b>Insurance:</b><br/>{_esc(_fmt(ev.get('insurance_guidance_en')))}", body),
    ]]
    ev_t = Table(ev_rows, colWidths=[110, 140, 160, 150])
    ev_t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f8e9")),
                              ("BOX", (0, 0), (-1, -1), 1, PRIMARY_GREEN), ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREEN),
                              ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    els.append(ev_t)

    # 6. footer --------------------------------------------------------------
    els.append(Spacer(1, 6))
    els.append(Paragraph(
        "<i>Generated by the FASAL SURAKSHA AI tool. Results are screening estimates from photos and entered data. "
        "This card does not decide any insurance claim or payout; those are determined by the insurer and the "
        "government under the scheme rules. For confirmation, contact your nearest Krishi Vigyan Kendra (KVK) "
        "or Agriculture Office.</i>", sub_style))

    doc.build(els)
    logger.info(f"Crop Health Card PDF generated: {pdf_path}")
    return pdf_path
