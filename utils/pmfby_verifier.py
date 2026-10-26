"""
====================================================================
FASAL SURAKSHA - utils/pmfby_verifier.py   (revised)
Crop-damage EVIDENCE helper for farmers / surveyors.

IMPORTANT - what this module no longer does
-------------------------------------------
The previous version computed an "eligibility" verdict and an INR payout
as  sum_insured x damage%.  That is not how PMFBY works:

* Pest / disease losses are part of the yield-loss cover, settled mainly
  on an AREA basis for the notified unit (crop-cutting experiments), not
  from one farmer's leaf photo.
* Individual-farm assessment applies only to localized calamities
  (hailstorm, landslide, inundation, cloud burst, natural fire) and
  post-harvest losses.
* Sums insured come from state-level scale-of-finance notifications,
  not from a hardcoded table.

So this module now only (a) organises photo / severity evidence, (b)
reports how complete that evidence is, and (c) points the farmer to the
right reporting channel.  It never states eligibility or a payout.
Check pmfby.gov.in and the current Operational Guidelines for rules.

`verify_pmfby_claim()` keeps its old name and arguments so existing
callers keep working.  Legacy keys `is_claim_eligible` and
`estimated_indemnity_payout_inr` are now None - update any UI code that
formats them (e.g. f"{x:,.2f}") to handle None / remove them.
====================================================================
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# Follow-up priority is about field-visit / advisory urgency only.
HIGH_SEVERITY = 50.0
MODERATE_SEVERITY = 20.0
LOW_CONFIDENCE = 60.0

INSURANCE_GUIDANCE_EN = (
    "Crop insurance is settled under the scheme's own rules, not by this app. "
    "Contact your insurance company, bank branch or local Agriculture Office, and see pmfby.gov.in "
    "for the current guidelines and claim-intimation timelines. For localized calamities such as "
    "hailstorm, landslide, inundation, cloud burst or natural fire, report quickly "
    "(the scheme sets short intimation windows)."
)
INSURANCE_GUIDANCE_HI = (
    "फसल बीमा के दावे योजना के नियमों के अनुसार तय होते हैं, इस ऐप से नहीं। "
    "अपनी बीमा कंपनी, बैंक शाखा या कृषि कार्यालय से संपर्क करें और pmfby.gov.in देखें।"
)


def verify_pmfby_claim(
    disease_name: str,
    severity_percent: float,
    predicted_yield_loss_percent: Optional[float] = None,
    farm_area_acres: float = 2.0,
    crop_name: str = "Tomato",
    farmer_id: Optional[str] = None,
    claim_id: Optional[str] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    survey_date: Optional[str] = None,
    model_confidence_percent: Optional[float] = None,
    photo_count: int = 1,
) -> Dict[str, Any]:
    """Build an evidence record. Does NOT decide eligibility or payout."""
    now = datetime.now(timezone.utc)
    geo_provided = gps_lat is not None and gps_lon is not None

    # ---- evidence completeness ------------------------------------
    missing: List[str] = []
    if not geo_provided:
        missing.append("GPS location (field photos should be geotagged)")
    if not farmer_id:
        missing.append("Farmer / insurance ID")
    if not survey_date:
        missing.append("Survey date")
    if photo_count < 3:
        missing.append("At least 3 photos (leaf close-up, plant, whole field)")
    if model_confidence_percent is not None and model_confidence_percent < LOW_CONFIDENCE:
        missing.append("Clearer photo or expert confirmation (low AI confidence)")
    evidence_status = "COMPLETE" if not missing else "INCOMPLETE"

    # ---- follow-up priority (advisory / field-visit urgency) --------
    if severity_percent >= HIGH_SEVERITY:
        priority, color, queue = "HIGH - arrange field visit / expert advice soon", "red", "Urgent field visit"
    elif severity_percent >= MODERATE_SEVERITY:
        priority, color, queue = "MEDIUM - treat now and document progress", "orange", "Advisory + documentation"
    else:
        priority, color, queue = "LOW - monitor and treat early", "green", "Advisory only"

    recommendation = (
        "Record dated, geotagged photos of the affected leaves, plants and whole field before and after treatment. "
        "Keep input purchase bills. Report to the insurer / bank / Agriculture Office through the official channel. "
        "This app's severity number is an estimate from one leaf image and is not a loss assessment."
    )

    result = {
        "claim_id": claim_id or f"EVD-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "farmer_id": farmer_id,  # None if not provided - never invented
        "crop_name": crop_name,
        "disease_detected": disease_name,
        "farm_area_acres": farm_area_acres,
        "leaf_severity_percent": round(float(severity_percent), 2),
        "predicted_yield_loss_percent": None if predicted_yield_loss_percent is None else round(float(predicted_yield_loss_percent), 2),
        "model_confidence_percent": model_confidence_percent,
        "evidence_status": evidence_status,
        "evidence_missing": missing,
        "follow_up_priority": priority,
        "risk_badge_color": color,
        "audit_queue": queue,
        "surveyor_recommendation": recommendation,
        "insurance_guidance_en": INSURANCE_GUIDANCE_EN,
        "insurance_guidance_hi": INSURANCE_GUIDANCE_HI,
        "audit_metadata": {
            "evaluation_timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "survey_date": survey_date,
            "geo_coordinates": f"{gps_lat:.5f}, {gps_lon:.5f}" if geo_provided else None,
            "geo_provided": geo_provided,  # coordinates were supplied; NOT independently verified
            "image_integrity_checked": False,
        },
        # ---- legacy keys, intentionally neutral -----------------------
        "is_claim_eligible": None,
        "estimated_indemnity_payout_inr": None,
        "triage_priority": priority,
    }
    logger.info(f"Evidence record {result['claim_id']}: status={evidence_status}, priority={color}")
    return result


# Clearer alias for new code
build_evidence_report = verify_pmfby_claim
