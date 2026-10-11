"""
====================================================================
FASAL SURAKSHA - utils/dosage_calculator.py
Smart Treatment & Dosage Recommendation Engine.
Outputs exact chemical and organic remedies, active ingredients,
and application mix ratios (mL/L or g/L) scaled dynamically to
the computed Disease Severity Index (DSI).
====================================================================
"""

from typing import Dict, Any, List
from loguru import logger

# Disease Registry mapped with ICAR & Agronomic Guidelines
DISEASE_REMEDY_DATABASE: Dict[str, Dict[str, Any]] = {
    "Tomato_Late_Blight": {
        "crop": "Tomato",
        "disease_name_en": "Late Blight",
        "disease_name_hi": "पछेती झुलसा रोग (Late Blight)",
        "causal_organism": "Phytophthora infestans (Oomycete)",
        "organic_remedy": {
            "name_en": "Copper Oxychloride + Trichoderma viride",
            "name_hi": "कॉपर ऑक्सीक्लोराइड + ट्राइकोडर्मा विरिडे",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 7,
            "instructions_en": "Spray thoroughly under leaves during morning hours. Repeat every 7-10 days.",
            "instructions_hi": "सुबह के समय पत्तियों की निचली सतह पर छिड़काव करें। हर 7-10 दिनों में दोहराएं।"
        },
        "chemical_remedy": {
            "name_en": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ)",
            "name_hi": "मेटालेक्सिल 8% + मैंकोजेब 64% डब्ल्यूपी (रिडोमिल)",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 10,
            "instructions_en": "Dissolve in clean water and apply with hollow cone nozzle at first appearance.",
            "instructions_hi": "साफ पानी में घोलकर पहले लक्षण दिखाई देते ही हॉलो कोन नोजल से छिड़कें।"
        },
        "safety_instructions_en": "Wear protective gear. Keep 14 days waiting period before harvesting.",
        "safety_instructions_hi": "सुरक्षात्मक किट पहनें। कटाई से पहले 14 दिन का अंतराल रखें।"
    },
    "Tomato_Early_Blight": {
        "crop": "Tomato",
        "disease_name_en": "Early Blight",
        "disease_name_hi": "अगेती झुलसा रोग (Early Blight)",
        "causal_organism": "Alternaria solani (Fungus)",
        "organic_remedy": {
            "name_en": "Neem Oil 10,000 PPM + Pseudomonas fluorescens",
            "name_hi": "नीम का तेल 10,000 पीपीएम + स्यूडोमोनास",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 8,
            "instructions_en": "Mix with mild liquid surfactant. Spray when air humidity exceeds 70%.",
            "instructions_hi": "हल्के सर्फेक्टेंट के साथ मिलाएं। हवा में नमी 70% से अधिक होने पर छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC (Amistar Top)",
            "name_hi": "एज़ोक्सीस्ट्रोबिन 18.2% + डिफेनोकोनाज़ोल 11.4% एससी",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 12,
            "instructions_en": "Foliar application covering canopy uniformly.",
            "instructions_hi": "पत्तियों और पौधों के ऊपरी हिस्से पर समान रूप से छिड़काव करें।"
        },
        "safety_instructions_en": "Avoid water run-off. 7 days pre-harvest interval.",
        "safety_instructions_hi": "दवा बहने न दें। कटाई से 7 दिन पूर्व प्रयोग बंद करें।"
    },
    "Potato_Late_Blight": {
        "crop": "Potato",
        "disease_name_en": "Late Blight",
        "disease_name_hi": "आलू पछेती झुलसा (Potato Late Blight)",
        "causal_organism": "Phytophthora infestans",
        "organic_remedy": {
            "name_en": "Bordeaux Mixture (1%)",
            "name_hi": "बोर्डो मिश्रण (1%)",
            "base_ratio_ml_per_liter": 10.0,
            "application_interval_days": 7,
            "instructions_en": "Apply freshly prepared Bordeaux mixture on overcast days.",
            "instructions_hi": "बादल छाए रहने वाले दिनों में ताजा तैयार बोर्डो मिश्रण छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Cymoxanil 8% + Mancozeb 64% WP",
            "name_hi": "साइमोक्सानिल 8% + मैंकोजेब 64% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 8,
            "instructions_en": "High protective systemic efficacy. Use 500L water per hectare.",
            "instructions_hi": "उच्च प्रणालीगत प्रभाव। प्रति हेक्टेयर 500 लीटर पानी का उपयोग करें।"
        },
        "safety_instructions_en": "Harmful to aquatic life. Do not apply near water bodies.",
        "safety_instructions_hi": "जलीय जीवों के लिए हानिकारक। जल स्रोतों के पास न छिड़कें।"
    },
    "Potato_Early_Blight": {
        "crop": "Potato",
        "disease_name_en": "Early Blight",
        "disease_name_hi": "आलू अगेती झुलसा (Potato Early Blight)",
        "causal_organism": "Alternaria solani",
        "organic_remedy": {
            "name_en": "Neem Oil 10,000 PPM + Trichoderma harzianum",
            "name_hi": "नीम तेल 10,000 पीपीएम + ट्राइकोडर्मा",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 7,
            "instructions_en": "Spray at first sign of concentric target spots.",
            "instructions_hi": "पत्तियों पर छल्लेदार धब्बे दिखते ही छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Mancozeb 75% WP (Dithane M-45)",
            "name_hi": "मैंकोजेब 75% डब्ल्यूपी (डाइथेन एम-45)",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 10,
            "instructions_en": "Uniform foliar application covering both foliage surfaces.",
            "instructions_hi": "पत्तियों की दोनों सतहों पर समान रूप से छिड़कें।"
        },
        "safety_instructions_en": "Keep 10 days pre-harvest safety interval.",
        "safety_instructions_hi": "कटाई से 10 दिन पहले छिड़काव बंद करें।"
    },
    "Apple_Apple_scab": {
        "crop": "Apple",
        "disease_name_en": "Apple Scab",
        "disease_name_hi": "सेब का स्केब रोग (Apple Scab)",
        "causal_organism": "Venturia inaequalis",
        "organic_remedy": {
            "name_en": "Liquid Lime Sulfur 28%",
            "name_hi": "तरल चूना गंधक 28%",
            "base_ratio_ml_per_liter": 5.0,
            "application_interval_days": 10,
            "instructions_en": "Dormant to pink bud stage protective application.",
            "instructions_hi": "कली खिलने से पहले सुरक्षात्मक छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Difenoconazole 25% EC (Score)",
            "name_hi": "डिफेनोकोनाज़ोल 25% ईसी (स्कोर)",
            "base_ratio_ml_per_liter": 0.5,
            "application_interval_days": 14,
            "instructions_en": "Post-infection curative systemic spray.",
            "instructions_hi": "रोग लक्षण दिखने पर तुरंत उपचारात्मक छिड़काव।"
        },
        "safety_instructions_en": "Wear protective suit. 14 days pre-harvest interval.",
        "safety_instructions_hi": "सुरक्षात्मक सूट पहनें। 14 दिन का अंतराल रखें।"
    },
    "Cotton_Bacterial_Blight": {
        "crop": "Cotton",
        "disease_name_en": "Bacterial Blight / Black Arm",
        "disease_name_hi": "कपास का जीवाणु झुलसा / काला हाथ रोग",
        "causal_organism": "Xanthomonas citri pv. malvacearum",
        "organic_remedy": {
            "name_en": "Pseudomonas fluorescens 1% WP",
            "name_hi": "स्यूडोमोनास फ्लोरोसेंस 1% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 8,
            "instructions_en": "Bio-bactericide spray at square formation stage.",
            "instructions_hi": "फूल-डोडियां बनते समय जैविक जीवाणुनाशी का छिड़काव।"
        },
        "chemical_remedy": {
            "name_en": "Streptocycline 90:10 + Copper Oxychloride 50% WP",
            "name_hi": "स्ट्रेप्टोसाइक्लिन + कॉपर ऑक्सीक्लोराइड 50% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 0.1,
            "application_interval_days": 12,
            "instructions_en": "Dissolve 6g Streptocycline + 300g COC in 200L water per acre.",
            "instructions_hi": "6 ग्राम स्ट्रेप्टोसाइक्लिन + 300 ग्राम सीओसी 200 लीटर पानी में।"
        },
        "safety_instructions_en": "Toxic if swallowed. Wear respirator during mixing.",
        "safety_instructions_hi": "मिश्रण बनाते समय मास्क अवश्य पहनें।"
    },
    "Rice_Blast": {
        "crop": "Rice/Paddy",
        "disease_name_en": "Rice Blast",
        "disease_name_hi": "धान का झोंका रोग (Rice Blast)",
        "causal_organism": "Magnaporthe oryzae",
        "organic_remedy": {
            "name_en": "Cow Urine (Gau Mutra) fermented solution + Neem extract",
            "name_hi": "गोमूत्र अर्क + नीम का काढ़ा",
            "base_ratio_ml_per_liter": 50.0,
            "application_interval_days": 5,
            "instructions_en": "Spray in early morning at tillering stage.",
            "instructions_hi": "कल्ले फूटने की अवस्था में सुबह जल्दी छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Tricyclazole 75% WP (Baan / Beam)",
            "name_hi": "ट्राइसाइक्लाज़ोल 75% डब्ल्यूपी (बान/बीम)",
            "base_ratio_ml_per_liter": 0.6,
            "application_interval_days": 14,
            "instructions_en": "Systemic fungicide targeting neck and leaf blast.",
            "instructions_hi": "गर्दन और पत्ती झोंका रोग के लिए प्रभावी प्रणालीगत कवकनाशी।"
        },
        "safety_instructions_en": "Use protective mask during spraying.",
        "safety_instructions_hi": "छिड़काव के समय मास्क और चश्मा अवश्य पहनें।"
    },
    "Wheat_Rust": {
        "crop": "Wheat",
        "disease_name_en": "Brown/Yellow Stripe Rust",
        "disease_name_hi": "गेहूं का रतुआ/गेरुआ रोग (Wheat Rust)",
        "causal_organism": "Puccinia striiformis",
        "organic_remedy": {
            "name_en": "Sulfur Dusting (80% WDG)",
            "name_hi": "सल्फर/गंधक (80% डब्लूडीजी)",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 10,
            "instructions_en": "Apply during calm wind conditions.",
            "instructions_hi": "हवा शांत होने पर पत्तियों पर छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Propiconazole 25% EC (Tilt)",
            "name_hi": "प्रोपिकोनाज़ोल 25% ईसी (टिल्ट)",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 15,
            "instructions_en": "Dissolve in 200 liters of water per acre.",
            "instructions_hi": "प्रति एकड़ 200 लीटर पानी में मिलाकर स्प्रे करें।"
        },
        "safety_instructions_en": "Highly inflammable in concentrate. Store away from heat.",
        "safety_instructions_hi": "आग और धूप से दूर रखें।"
    },
    "Corn_Common_Rust": {
        "crop": "Corn/Maize",
        "disease_name_en": "Common Rust",
        "disease_name_hi": "मक्का का सामान्य रतुआ रोग (Corn Rust)",
        "causal_organism": "Puccinia sorghi",
        "organic_remedy": {
            "name_en": "Bacillus subtilis formulation",
            "name_hi": "बैसिलस सबटिलिस जैविक कीटनाशक",
            "base_ratio_ml_per_liter": 4.0,
            "application_interval_days": 7,
            "instructions_en": "Preventative bio-fungicide spray.",
            "instructions_hi": "रोकथाम हेतु जैविक कवकनाशी का छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Pyraclostrobin 20% WG",
            "name_hi": "पाइराक्लोस्ट्रोबिन 20% डब्ल्यूजी",
            "base_ratio_ml_per_liter": 1.2,
            "application_interval_days": 14,
            "instructions_en": "Apply at the onset of pustules formation.",
            "instructions_hi": "पत्तियों पर फफोले बनते ही तुरंत छिड़कें।"
        },
        "safety_instructions_en": "Do not graze livestock in treated area for 14 days.",
        "safety_instructions_hi": "उपचारित खेत में 14 दिनों तक मवेशियों को न चरने दें।"
    },
    "Healthy_Crop": {
        "crop": "Healthy Leaf",
        "disease_name_en": "Healthy Crop - No Infection Detected",
        "disease_name_hi": "स्वस्थ फसल - कोई रोग नहीं पाया गया",
        "causal_organism": "None",
        "organic_remedy": {
            "name_en": "Seaweed Liquid Extract Bio-stimulant",
            "name_hi": "समुद्री शैवाल का तरल अर्क (बायो-उत्तेजक)",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 21,
            "instructions_en": "Nutritional maintenance spray to enhance immunity and chlorophyll.",
            "instructions_hi": "रोग प्रतिरोधक क्षमता और हरापन बढ़ाने के लिए पोषण स्प्रे।"
        },
        "chemical_remedy": {
            "name_en": "NPK 19:19:19 Micronutrient Foliar Spray",
            "name_hi": "एनपीके 19:19:19 घुलनशील उर्वरक",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 30,
            "instructions_en": "Balanced nutrient boost.",
            "instructions_hi": "संतुलित फसल विकास हेतु छिड़काव।"
        },
        "safety_instructions_en": "Standard agricultural handling.",
        "safety_instructions_hi": "सामान्य कृषि सुरक्षा नियमों का पालन करें।"
    }
}


def calculate_dynamic_dosage(
    disease_key: str,
    severity_percent: float,
    farm_area_acres: float = 1.0,
    spray_tank_liters: float = 15.0
) -> Dict[str, Any]:
    """
    Dynamically scales the required dosage based on severity index:
    - Mild (<20%): Standard baseline dose
    - Moderate (20-50%): 1.25x scaling factor + reduced application interval
    - Severe (>50%): 1.50x scaling factor + urgent repeat schedule
    """
    # Normalize lookup key
    matched_key = "Healthy_Crop"
    if "healthy" in disease_key.lower():
        matched_key = "Healthy_Crop"
    else:
        for k in DISEASE_REMEDY_DATABASE.keys():
            if k.lower() in disease_key.lower() or disease_key.lower() in k.lower():
                matched_key = k
                break
        if matched_key == "Healthy_Crop" and disease_key != "Healthy_Crop":
            # fallback to generic tomato blight recommendation if unknown disease
            matched_key = "Tomato_Late_Blight"

    info = DISEASE_REMEDY_DATABASE[matched_key]

    # Severity scaling multiplier
    if severity_percent < 20.0:
        multiplier = 1.0
        urgency = "LOW / PREVENTATIVE (सामान्य)"
        schedule_adj = 0
    elif 20.0 <= severity_percent < 50.0:
        multiplier = 1.25
        urgency = "MODERATE / CURATIVE (मध्यम)"
        schedule_adj = -2
    else:
        multiplier = 1.50
        urgency = "CRITICAL / EMERGENCY RESCUE (अति गंभीर)"
        schedule_adj = -4

    # Calculate calibrated doses
    organic = info["organic_remedy"].copy()
    chemical = info["chemical_remedy"].copy()

    org_per_liter = round(organic["base_ratio_ml_per_liter"] * multiplier, 2)
    chem_per_liter = round(chemical["base_ratio_ml_per_liter"] * multiplier, 2)

    org_per_tank = round(org_per_liter * spray_tank_liters, 1)
    chem_per_tank = round(chem_per_liter * spray_tank_liters, 1)

    # Standard water volume: ~150-200 Liters per Acre (approx 10-13 knapsack tanks)
    tanks_per_acre = round(200.0 / spray_tank_liters, 1)
    total_chem_per_acre = round(chem_per_liter * 200.0 * farm_area_acres, 1)
    total_org_per_acre = round(org_per_liter * 200.0 * farm_area_acres, 1)

    org_interval = max(3, organic["application_interval_days"] + schedule_adj)
    chem_interval = max(4, chemical["application_interval_days"] + schedule_adj)

    response = {
        "disease_key": matched_key,
        "crop": info["crop"],
        "disease_name_en": info["disease_name_en"],
        "disease_name_hi": info["disease_name_hi"],
        "causal_organism": info["causal_organism"],
        "severity_level": "Mild" if severity_percent < 20 else ("Moderate" if severity_percent < 50 else "Severe"),
        "severity_percent": round(severity_percent, 2),
        "urgency_level": urgency,
        "multiplier": multiplier,
        "farm_parameters": {
            "farm_area_acres": farm_area_acres,
            "spray_tank_volume_liters": spray_tank_liters,
            "approx_tanks_per_acre": tanks_per_acre
        },
        "organic_plan": {
            "name_en": organic["name_en"],
            "name_hi": organic["name_hi"],
            "concentration_per_liter": f"{org_per_liter} mL (or g) / Liter water",
            "knapsack_tank_dose": f"{org_per_tank} mL (or g) for {spray_tank_liters}L tank",
            "total_farm_requirement": f"{total_org_per_acre} mL (or g) for {farm_area_acres} Acre(s)",
            "interval_days": org_interval,
            "instructions_en": organic["instructions_en"],
            "instructions_hi": organic["instructions_hi"]
        },
        "chemical_plan": {
            "name_en": chemical["name_en"],
            "name_hi": chemical["name_hi"],
            "concentration_per_liter": f"{chem_per_liter} mL (or g) / Liter water",
            "knapsack_tank_dose": f"{chem_per_tank} mL (or g) for {spray_tank_liters}L tank",
            "total_farm_requirement": f"{total_chem_per_acre} mL (or g) for {farm_area_acres} Acre(s)",
            "interval_days": chem_interval,
            "instructions_en": chemical["instructions_en"],
            "instructions_hi": chemical["instructions_hi"]
        },
        "safety_instructions_en": info["safety_instructions_en"],
        "safety_instructions_hi": info["safety_instructions_hi"]
    }
    logger.info(f"Dosage calculated for {matched_key} at {severity_percent}% severity (Multiplier: {multiplier}x)")
    return response
