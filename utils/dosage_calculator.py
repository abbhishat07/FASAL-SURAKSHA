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
try:
    from loguru import logger
except (ImportError, ModuleNotFoundError):
    import logging
    logger = logging.getLogger("fasal_suraksha")

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
    "Grape_Black_rot": {
        "crop": "Grape",
        "disease_name_en": "Black Rot",
        "disease_name_hi": "अंगूर का काला सड़न रोग (Black Rot)",
        "causal_organism": "Guignardia bidwellii",
        "organic_remedy": {
            "name_en": "Bordeaux Mixture 1% or Copper Hydroxide 77% WP",
            "name_hi": "बोर्डो मिश्रण 1% या कॉपर हाइड्रॉक्साइड 77% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 8,
            "instructions_en": "Apply thoroughly on grape bunches and both foliage surfaces before bloom.",
            "instructions_hi": "फूल आने से पहले पत्तियों की दोनों सतहों और गुच्छों पर अच्छी तरह छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Myclobutanil 10% WP (Systhane) + Mancozeb 75% WP",
            "name_hi": "माइक्लोब्यूटानिल 10% डब्ल्यूपी + मैंकोजेब 75% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 1.5,
            "application_interval_days": 10,
            "instructions_en": "Target systemic spray at early shoot growth stage to prevent mummification.",
            "instructions_hi": "अंगूर को सिकुड़ने से बचाने के लिए नई शाखाएं निकलते समय छिड़काव करें।"
        },
        "safety_instructions_en": "Do not spray within 14 days of harvest. Wear eye protection.",
        "safety_instructions_hi": "कटाई से 14 दिन पहले छिड़काव बंद करें। आंखों की सुरक्षा रखें।"
    },
    "Grape_Esca": {
        "crop": "Grape",
        "disease_name_en": "Esca (Black Measles / Tiger-Stripe)",
        "disease_name_hi": "अंगूर का एस्का / ब्लैक मीजल्स रोग (Esca)",
        "causal_organism": "Phaeomoniella chlamydospora & Fomitiporia mediterranea",
        "organic_remedy": {
            "name_en": "Trichoderma atroviride + Pruning Wound Sealant",
            "name_hi": "ट्राइकोडर्मा अट्रोविरिड + छंटाई घाव लेप",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 12,
            "instructions_en": "Paint pruning cuts promptly; apply foliar bio-fungicide to protect vascular tissues.",
            "instructions_hi": "छंटाई के तुरंत बाद घावों पर लेप लगाएं और संवहनी ऊतकों की सुरक्षा हेतु स्प्रे करें।"
        },
        "chemical_remedy": {
            "name_en": "Thiophanate-Methyl 70% WP + Fosetyl-Al 80% WP (Aliette)",
            "name_hi": "थियोफैनेट-मिथाइल 70% डब्ल्यूपी + फॉसिटिल-एल 80% डब्ल्यूपी (एलियट)",
            "base_ratio_ml_per_liter": 1.5,
            "application_interval_days": 14,
            "instructions_en": "Systemic upward and downward translocation to arrest vine decline and tiger-striping.",
            "instructions_hi": "बाघ जैसी धारियों और बेल के सूखने को रोकने के लिए प्रणालीगत छिड़काव करें।"
        },
        "safety_instructions_en": "Avoid inhaling spray mist. Keep 21 days pre-harvest safety interval.",
        "safety_instructions_hi": "स्प्रे धुंध को सांस में न लें। कटाई से 21 दिन पहले प्रयोग बंद करें।"
    },
    "Grape_Leaf_blight": {
        "crop": "Grape",
        "disease_name_en": "Leaf Blight (Isariopsis Spot)",
        "disease_name_hi": "अंगूर का पत्ती झुलसा रोग (Leaf Blight)",
        "causal_organism": "Pseudocercospora cladosporioides",
        "organic_remedy": {
            "name_en": "Potassium Bicarbonate + Wettable Sulfur 80% WDG",
            "name_hi": "पोटेशियम बाइकार्बोनेट + घुलनशील गंधक 80%",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 7,
            "instructions_en": "Foliar canopy spray at sunrise to curb angular leaf necrosis expansion.",
            "instructions_hi": "कोणीय धब्बों के फैलाव को रोकने के लिए सूर्योदय के समय छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Azoxystrobin 11% + Tebuconazole 18.3% w/w SC (Custodia)",
            "name_hi": "एज़ोक्सीस्ट्रोबिन 11% + टेबुकोनाज़ोल 18.3% एससी",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 12,
            "instructions_en": "Dual broad-spectrum strobilurin-triazole curative foliar spray.",
            "instructions_hi": "रोग प्रतिरोधक दोहरा उपचारात्मक छिड़काव पत्तियों पर करें।"
        },
        "safety_instructions_en": "Toxic to bees and aquatic organisms. Spray late evening if hives nearby.",
        "safety_instructions_hi": "मधुमक्खियों के लिए विषैला। मधुमक्खी के छत्तों के पास देर शाम छिड़कें।"
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
    "Orange_Haunglongbing": {
        "crop": "Orange (Citrus)",
        "disease_name_en": "Huanglongbing (Citrus Greening)",
        "disease_name_hi": "सिट्रस ग्रीनिंग / हुआंगलोंगबिंग (Citrus Greening)",
        "causal_organism": "Candidatus Liberibacter asiaticus (Bacterium, vectored by Diaphorina citri)",
        "organic_remedy": {
            "name_en": "Neem Seed Kernel Extract (NSKE 5%) + Micronutrient Foliar (Zn+Fe+Mn)",
            "name_hi": "नीम बीज अर्क (एनएसकेई 5%) + सूक्ष्म पोषक तत्व (जिंक+आयरन+मैंगनीज)",
            "base_ratio_ml_per_liter": 5.0,
            "application_interval_days": 10,
            "instructions_en": "Spray foliage thoroughly to suppress psyllid vector. Apply zinc and iron micronutrients to correct greening chlorosis.",
            "instructions_hi": "साइला कीट नियंत्रण हेतु छिड़कें। पत्तों के पीलेपन को ठीक करने के लिए सूक्ष्म पोषक तत्व दें।"
        },
        "chemical_remedy": {
            "name_en": "Imidacloprid 17.8% SL (0.5 mL/L) or Thiamethoxam 25% WG (0.4 g/L)",
            "name_hi": "इमिडाक्लोप्रिड 17.8% एसएल (0.5 मिली/ली) अथवा थायमेथॉक्सम 25% डब्ल्यूजी",
            "base_ratio_ml_per_liter": 0.5,
            "application_interval_days": 14,
            "instructions_en": "Target Asian Citrus Psyllid during new flush growth. Drench tree root zone or trunk application.",
            "instructions_hi": "नई कोपलें आने पर सिट्रस साइला को नियंत्रित करने हेतु छिड़कें। तने व जड़ों के पास भी डालें।"
        },
        "safety_instructions_en": "Harmful to honeybees. Do not apply during citrus blossom/flowering.",
        "safety_instructions_hi": "मधुमक्खियों के लिए हानिकारक। फूलों के खिलने के समय छिड़काव न करें।"
    },
    "Peach_Bacterial_spot": {
        "crop": "Peach",
        "disease_name_en": "Bacterial Spot",
        "disease_name_hi": "आड़ू जीवाणु धब्बा रोग (Bacterial Spot)",
        "causal_organism": "Xanthomonas arboricola pv. pruni",
        "organic_remedy": {
            "name_en": "Copper Hydroxide 53.8% DF + Bacillus subtilis QST 713",
            "name_hi": "कॉपर हाइड्रॉक्साइड 53.8% + बैसिलस सबटिलिस",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 8,
            "instructions_en": "Dormant to shuck-split stage application. Avoid phytotoxicity under slow drying conditions.",
            "instructions_hi": "सुषुप्तावस्था से पत्तियां निकलने तक छिड़कें। नम मौसम में अत्यधिक छिड़काव से बचें।"
        },
        "chemical_remedy": {
            "name_en": "Oxytetracycline 200 PPM + Mancozeb 75% WP (2 g/L)",
            "name_hi": "ऑक्सीटेट्रासाइक्लिन 200 पीपीएम + मैंकोजेब 75% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 10,
            "instructions_en": "Apply at 7-10 day intervals starting at petal fall during warm, wet conditions.",
            "instructions_hi": "पंखुड़ियां गिरने के बाद गर्म व नम मौसम में 7-10 दिनों के अंतराल पर छिड़कें।"
        },
        "safety_instructions_en": "Observe 21-day pre-harvest interval on stone fruit crops.",
        "safety_instructions_hi": "फल तुड़ाई से 21 दिन पूर्व रासायनिक छिड़काव बंद करें।"
    },
    "Squash_Powdery_mildew": {
        "crop": "Squash / Cucurbit",
        "disease_name_en": "Powdery Mildew",
        "disease_name_hi": "कद्दू वर्गीय चूर्णी फफूंद (Powdery Mildew)",
        "causal_organism": "Podosphaera xanthii (Fungus)",
        "organic_remedy": {
            "name_en": "Potassium Bicarbonate (3 g/L) + Neem Oil 10,000 PPM",
            "name_hi": "पोटेशियम बाइकार्बोनेट (3 ग्राम/ली) + नीम तेल 10,000 पीपीएम",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 7,
            "instructions_en": "Coat both upper and lower leaf surfaces. Potassium bicarbonate alters leaf pH to destroy fungal mycelium.",
            "instructions_hi": "पत्तियों की दोनों सतहों पर छिड़कें। यह फफूंद के कवकजाल को नष्ट करता है।"
        },
        "chemical_remedy": {
            "name_en": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC or Dinocap 48% EC",
            "name_hi": "एज़ोक्सीस्ट्रोबिन + डिफेनोकोनाज़ोल एससी अथवा डाइनोकैप 48% ईसी",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 12,
            "instructions_en": "Apply as soon as talcum-powder like white spots first appear on crown leaves.",
            "instructions_hi": "पत्तियों पर सफेद पाउडर जैसे धब्बे दिखते ही तुरंत छिड़काव करें।"
        },
        "safety_instructions_en": "Do not exceed 3 applications per season to prevent strobilurin resistance.",
        "safety_instructions_hi": "प्रति मौसम 3 से अधिक छिड़काव न करें ताकि दवा का असर बना रहे।"
    },
    "Strawberry_Leaf_scorch": {
        "crop": "Strawberry",
        "disease_name_en": "Leaf Scorch",
        "disease_name_hi": "स्ट्रॉबेरी पत्ती झुलसा (Leaf Scorch)",
        "causal_organism": "Diplocarpon earlianum (Fungus)",
        "organic_remedy": {
            "name_en": "Copper Oxychloride 50% WP (2 g/L) + Trichoderma viride",
            "name_hi": "कॉपर ऑक्सीक्लोराइड 50% + ट्राइकोडर्मा विरिडे",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 8,
            "instructions_en": "Remove and destroy severely scorched outer leaves before spraying.",
            "instructions_hi": "छिड़काव से पहले अत्यधिक झुलसी हुई पुरानी पत्तियों को तोड़कर नष्ट कर दें।"
        },
        "chemical_remedy": {
            "name_en": "Thiophanate-Methyl 70% WP (1 g/L) or Captan 50% WP (2.5 g/L)",
            "name_hi": "थायोफैनेट-मिथाइल 70% डब्ल्यूपी अथवा कैप्टन 50% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 1.5,
            "application_interval_days": 10,
            "instructions_en": "Spray early morning ensuring thorough wetting of runner foliage.",
            "instructions_hi": "सुबह के समय सभी पत्तियों पर समान रूप से छिड़काव करें।"
        },
        "safety_instructions_en": "7 days pre-harvest interval before berry picking.",
        "safety_instructions_hi": "स्ट्रॉबेरी तोड़ने से 7 दिन पहले छिड़काव रोक दें।"
    },
    "Tomato_Spider_mites": {
        "crop": "Tomato",
        "disease_name_en": "Two-Spotted Spider Mites",
        "disease_name_hi": "लाल मकड़ी / माइट्स प्रकोप (Spider Mites)",
        "causal_organism": "Tetranychus urticae (Acarina)",
        "organic_remedy": {
            "name_en": "Neem Seed Kernel Extract (NSKE 5%) + Predatory Mites",
            "name_hi": "नीम का काढ़ा (एनएसकेई 5%) + मित्र कीट",
            "base_ratio_ml_per_liter": 5.0,
            "application_interval_days": 6,
            "instructions_en": "Spray undersides of leaves with high pressure to dislodge webbing and nymphs.",
            "instructions_hi": "पत्तियों के नीचे की सतह पर तेज दबाव से छिड़कें ताकि जाले नष्ट हो सकें।"
        },
        "chemical_remedy": {
            "name_en": "Spiromesifen 22.9% SC (1 mL/L) or Abamectin 1.9% EC (0.75 mL/L)",
            "name_hi": "स्पाइरोमेसिफेन 22.9% एससी (1 मिली/ली) अथवा एबामेक्टिन 1.9% ईसी",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 10,
            "instructions_en": "Target both eggs and adult mites. Alternate with Propargite 57% EC if infestation is severe.",
            "instructions_hi": "अंडों और वयस्क कीटों दोनों पर प्रभावी। तेज प्रकोप में प्रोपरगाइट के साथ बदल-बदल कर छिड़कें।"
        },
        "safety_instructions_en": "Extremely toxic to fish. Keep away from irrigation ponds.",
        "safety_instructions_hi": "मछलियों के लिए अत्यधिक विषैला। जल स्रोतों से दूर रखें।"
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease_name_en": "Bacterial Spot",
        "disease_name_hi": "टमाटर जीवाणु धब्बा (Bacterial Spot)",
        "causal_organism": "Xanthomonas campestris pv. vesicatoria",
        "organic_remedy": {
            "name_en": "Pseudomonas fluorescens 1% WP (5 g/L) + Copper Oxychloride",
            "name_hi": "स्यूडोमोनास फ्लोरोसेंस + कॉपर ऑक्सीक्लोराइड",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 7,
            "instructions_en": "Foliar spray to prevent bacterial spread through rain splash.",
            "instructions_hi": "बारिश के छींटों से जीवाणु के फैलाव को रोकने के लिए छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Copper Hydroxide 77% WP (2 g/L) + Streptocycline 100 PPM (1 g / 10L)",
            "name_hi": "कॉपर हाइड्रॉक्साइड + स्ट्रेप्टोसाइक्लिन (1 ग्राम प्रति 10 लीटर पानी)",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 10,
            "instructions_en": "Tank mix copper with streptocycline antibiotic to overcome copper tolerance.",
            "instructions_hi": "कॉपर और स्ट्रेप्टोसाइक्लिन को मिलाकर छिड़कें।"
        },
        "safety_instructions_en": "Do not work in wet fields to avoid spreading bacterial inoculum.",
        "safety_instructions_hi": "गीले खेत में काम न करें जिससे जीवाणु न फैलें।"
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease_name_en": "Septoria Leaf Spot",
        "disease_name_hi": "सेप्टोरिया पत्ती धब्बा रोग",
        "causal_organism": "Septoria lycopersici",
        "organic_remedy": {
            "name_en": "Trichoderma harzianum (4 g/L) + Copper Hydroxide",
            "name_hi": "ट्राइकोडर्मा हरजिएनम + कॉपर कवकनाशी",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 8,
            "instructions_en": "Prune lower infected leaves first, then spray remaining canopy.",
            "instructions_hi": "निचली संक्रमित पत्तियों को हटाकर शेष पौधों पर छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Chlorothalonil 75% WP (2 g/L) or Mancozeb 75% WP (2.5 g/L)",
            "name_hi": "क्लोरोथैलोनिल 75% डब्ल्यूपी अथवा मैंकोजेब 75% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 10,
            "instructions_en": "Begin preventive sprays when first fruit clusters set.",
            "instructions_hi": "फल लगने की शुरुआत से ही सुरक्षात्मक छिड़काव करें।"
        },
        "safety_instructions_en": "Wear eye protection. 7 days pre-harvest interval.",
        "safety_instructions_hi": "आंखों की सुरक्षा रखें। कटाई से 7 दिन पूर्व प्रयोग रोकें।"
    },
    "Tomato_Target_Spot": {
        "crop": "Tomato",
        "disease_name_en": "Target Spot",
        "disease_name_hi": "टारगेट स्पॉट रोग",
        "causal_organism": "Corynespora cassiicola",
        "organic_remedy": {
            "name_en": "Bacillus amyloliquefaciens + Neem Oil 10,000 PPM",
            "name_hi": "बैसिलस एमाइलोलिक्विफेशिएन्स + नीम तेल",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 8,
            "instructions_en": "Apply preventative bio-fungicide during warm humid spells.",
            "instructions_hi": "गर्म व उमस भरे मौसम में सुरक्षात्मक जैविक छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Pyraclostrobin 13.3% + Boscalid 25.2% WG (Pristine, 1 g/L)",
            "name_hi": "पाइराक्लोस्ट्रोबिन + बॉस्केलिड डब्ल्यूजी (1 ग्राम/ली)",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 12,
            "instructions_en": "Systemic and contact control for concentric necrotic target lesions.",
            "instructions_hi": "छल्लों वाले काले धब्बों को रोकने हेतु प्रणालीगत दवा का छिड़काव करें।"
        },
        "safety_instructions_en": "Rotate with different mode-of-action fungicides.",
        "safety_instructions_hi": "दवा का प्रभाव बनाए रखने हेतु रसायनों को बदल-बदल कर उपयोग करें।"
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease_name_en": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "disease_name_hi": "टमाटर पीला पर्ण कुंचन विषाणु (TYLCV)",
        "causal_organism": "Begomovirus (Whitefly-vectored by Bemisia tabaci)",
        "organic_remedy": {
            "name_en": "Yellow Sticky Traps (20/acre) + NSKE 5% + Roguing",
            "name_hi": "पीले चिपचिपे ट्रैप (20 प्रति एकड़) + नीम काढ़ा 5% + संक्रमित पौधे उखाड़ना",
            "base_ratio_ml_per_liter": 5.0,
            "application_interval_days": 6,
            "instructions_en": "Uproot and bury symptomatic stunted plants. Install yellow sticky traps to capture whiteflies.",
            "instructions_hi": "गंभीर रूप से मुड़े हुए पौधों को उखाड़कर दबा दें। सफेद मक्खी पकड़ने के लिए ट्रैप लगाएं।"
        },
        "chemical_remedy": {
            "name_en": "Thiamethoxam 25% WG (0.35 g/L) or Cyantraniliprole 10.26% OD (1.8 mL/L)",
            "name_hi": "थायमेथॉक्सम 25% डब्ल्यूजी (0.35 ग्राम/ली) अथवा साएंट्रानिलिप्रोल 10.26% ओडी",
            "base_ratio_ml_per_liter": 0.5,
            "application_interval_days": 10,
            "instructions_en": "Foliar and nursery drenching to kill whitefly vector carrying virus.",
            "instructions_hi": "वायरस फैलाने वाली सफेद मक्खी को मारने हेतु पत्तियों पर छिड़काव करें।"
        },
        "safety_instructions_en": "Spray early morning or late afternoon when whiteflies are stationary.",
        "safety_instructions_hi": "सुबह या शाम के समय छिड़कें जब कीट पत्तियों पर स्थिर हों।"
    },
    "Tomato_Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease_name_en": "Tomato Mosaic Virus (ToMV)",
        "disease_name_hi": "टमाटर मोज़ेक विषाणु (ToMV)",
        "causal_organism": "Tobamovirus (Mechanically transmitted)",
        "organic_remedy": {
            "name_en": "Skimmed Milk Spray (10%) + Disinfect tools in 10% TSP",
            "name_hi": "मलाई रहित दूध का छिड़काव (10%) + औजारों को टीएसपी से धोना",
            "base_ratio_ml_per_liter": 100.0,
            "application_interval_days": 7,
            "instructions_en": "Milk proteins bind virus particles and inhibit mechanical transmission between plants.",
            "instructions_hi": "दूध के प्रोटीन वायरस कणों को बांधकर अन्य पौधों में फैलने से रोकते हैं।"
        },
        "chemical_remedy": {
            "name_en": "Micronutrient Zinc-Boron Foliar + Carbendazim 50% WP (prevent secondary rot)",
            "name_hi": "सूक्ष्म पोषक तत्व जिंक-बोरॉन + कार्बेन्डाजिम 50% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 1.5,
            "application_interval_days": 12,
            "instructions_en": "Support plant vigor and prevent secondary opportunist fungal blights.",
            "instructions_hi": "पौधों को मजबूती देने और द्वितीयक फफूंद से बचाव हेतु छिड़कें।"
        },
        "safety_instructions_en": "Wash hands with soap and trisodium phosphate before handling seedlings.",
        "safety_instructions_hi": "पौधों को छूने से पहले साबुन से हाथ अच्छी तरह धोएं।"
    },
    "Corn_Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease_name_en": "Northern Leaf Blight",
        "disease_name_hi": "मक्का उत्तरी पत्ती झुलसा (Northern Leaf Blight)",
        "causal_organism": "Exserohilum turcicum (Fungus)",
        "organic_remedy": {
            "name_en": "Trichoderma viride (5 g/L) + Pseudomonas fluorescens",
            "name_hi": "ट्राइकोडर्मा विरिडे + स्यूडोमोनास फ्लोरोसेंस",
            "base_ratio_ml_per_liter": 4.0,
            "application_interval_days": 8,
            "instructions_en": "Spray canopy before tasseling. Incorporate crop residues deep into soil.",
            "instructions_hi": "मक्का में भुट्टा/नर मंजरी आने से पहले पत्तियों पर छिड़काव करें।"
        },
        "chemical_remedy": {
            "name_en": "Propiconazole 25% EC (1 mL/L) or Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
            "name_hi": "प्रोपिकोनाज़ोल 25% ईसी (1 मिली/ली) अथवा एज़ोक्सीस्ट्रोबिन + डिफेनोकोनाज़ोल",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 14,
            "instructions_en": "Apply when cigar-shaped elliptical lesions appear on lower third of crop canopy.",
            "instructions_hi": "निचली पत्तियों पर सिगार के आकार के लंबे धब्बे दिखते ही छिड़कें।"
        },
        "safety_instructions_en": "Do not apply within 30 days of maize grain or silage harvest.",
        "safety_instructions_hi": "कटाई से 30 दिन पहले रासायनिक छिड़काव बंद करें।"
    },
    "Pepper_Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease_name_en": "Bacterial Spot",
        "disease_name_hi": "शिमला मिर्च जीवाणु धब्बा (Bacterial Spot)",
        "causal_organism": "Xanthomonas euvesicatoria",
        "organic_remedy": {
            "name_en": "Copper Oxychloride 50% WP + Bacillus subtilis",
            "name_hi": "कॉपर ऑक्सीक्लोराइड + बैसिलस सबटिलिस",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 7,
            "instructions_en": "Spray at first sign of water-soaked spots on foliage.",
            "instructions_hi": "पत्तियों पर पानी जैसे धब्बे दिखते ही छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Copper Hydroxide 77% WP (2 g/L) + Kasugamycin 3% SL (2 mL/L)",
            "name_hi": "कॉपर हाइड्रॉक्साइड + कासुगामाइसिन 3% एसएल (2 मिली/ली)",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 10,
            "instructions_en": "Combines protective bactericide with systemic antibiotic control.",
            "instructions_hi": "जीवाणु नियंत्रण हेतु प्रणालीगत एंटीबायोटिक के साथ मिलाकर छिड़कें।"
        },
        "safety_instructions_en": "10 days pre-harvest interval before picking bell peppers.",
        "safety_instructions_hi": "शिमला मिर्च तोड़ने से 10 दिन पूर्व छिड़काव बंद करें।"
    },
    "Mango_Anthracnose": {
        "crop": "Mango",
        "disease_name_en": "Mango Anthracnose (Leaf Blight & Black Spot)",
        "disease_name_hi": "आम का एन्थ्रेक्नोज़ / काला धब्बा रोग (Anthracnose)",
        "causal_organism": "Colletotrichum gloeosporioides (Fungus)",
        "organic_remedy": {
            "name_en": "Copper Oxychloride 50% WP (3.0 g/L) + Trichoderma viride",
            "name_hi": "कॉपर ऑक्सीक्लोराइड 50% डब्ल्यूपी + ट्राइकोडर्मा विरिडे",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 10,
            "instructions_en": "Spray foliage thoroughly at first appearance of circular or irregular dark brown to black spots. Repeat during rainy/humid weather.",
            "instructions_hi": "पत्तियों पर काले-भूरे धब्बे दिखते ही छिड़काव करें। बारिश और उच्च आर्द्रता में दोबारा छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Carbendazim 50% WP (Bavistin, 1.0 g/L) or Azoxystrobin 23% SC (1.0 mL/L)",
            "name_hi": "कार्बेन्डाजिम 50% डब्ल्यूपी (बाविस्टिन, 1 ग्राम/ली) अथवा एज़ोक्सीस्ट्रोबिन 23% एससी",
            "base_ratio_ml_per_liter": 1.0,
            "application_interval_days": 14,
            "instructions_en": "Systemic fungicide spray at flowering and leaf flush stages to prevent blossom blight and foliage drop.",
            "instructions_hi": "फूल आने और नई पत्तियां निकलने पर प्रणालीगत कवकनाशी का छिड़काव करें।"
        },
        "safety_instructions_en": "Observe 14-day pre-harvest interval. Avoid spraying during peak honeybee pollinator activity.",
        "safety_instructions_hi": "तुड़ाई से 14 दिन पूर्व छिड़काव रोकें। परागण के समय मधुमक्खियों के बचाव हेतु दोपहर में छिड़काव न करें।"
    },
    "Mango_Bacterial_canker": {
        "crop": "Mango",
        "disease_name_en": "Mango Bacterial Canker / Black Spot",
        "disease_name_hi": "आम का जीवाणु कैंकर / ब्लैक स्पॉट (Bacterial Canker)",
        "causal_organism": "Xanthomonas campestris pv. mangiferaeindicae (Bacterium)",
        "organic_remedy": {
            "name_en": "Pseudomonas fluorescens (5 g/L) + Neem Oil 10,000 PPM",
            "name_hi": "स्यूडोमोनास फ्लोरोसेंस (5 ग्राम/ली) + नीम तेल 10,000 पीपीएम",
            "base_ratio_ml_per_liter": 4.0,
            "application_interval_days": 8,
            "instructions_en": "Spray on leaves showing water-soaked angular lesions. Disinfect pruning shears to prevent mechanical transmission.",
            "instructions_hi": "पानी जैसे कोणीय धब्बों पर छिड़कें। रोग फैलने से रोकने हेतु छंटाई वाले औजारों को विसंक्रमित करें।"
        },
        "chemical_remedy": {
            "name_en": "Streptocycline (1 g/10 L) + Copper Oxychloride 50% WP (3.0 g/L)",
            "name_hi": "स्ट्रेप्टोसाइक्लिन (1 ग्राम प्रति 10 लीटर) + कॉपर ऑक्सीक्लोराइड 50% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 12,
            "instructions_en": "Apply bactericide tank mix at first sign of angular black lesions with yellow halos on foliage and twigs.",
            "instructions_hi": "पीले घेरे वाले काले धब्बे दिखते ही जीवाणुनाशक घोल का अच्छी तरह छिड़काव करें।"
        },
        "safety_instructions_en": "Do not exceed recommended antibiotic concentration. Wear protective mask and gloves.",
        "safety_instructions_hi": "एंटीबायोटिक की अनुशंसित मात्रा से अधिक न डालें। मास्क और दस्ताने पहनें।"
    },
    "Guava_Canker": {
        "crop": "Guava",
        "disease_name_en": "Guava Canker & Scab",
        "disease_name_hi": "अमरूद का कैंकर एवं खुरंड रोग (Guava Canker)",
        "causal_organism": "Pestalotiopsis psidii (Fungus)",
        "organic_remedy": {
            "name_en": "Bordeaux Mixture (1%) or Copper Hydroxide 53.8% DF",
            "name_hi": "बोर्डो मिश्रण (1%) अथवा कॉपर हाइड्रॉक्साइड 53.8%",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 10,
            "instructions_en": "Spray immediately after winter pruning and during high monsoon humidity to protect young leaves and shoots.",
            "instructions_hi": "सर्दियों की छंटाई के बाद और मानसूनी नमी में नई कोपलों और पत्तियों पर छिड़कें।"
        },
        "chemical_remedy": {
            "name_en": "Copper Oxychloride 50% WP (Blitox 50, 3.0 g/L) or Difenoconazole 25% EC (0.5 mL/L)",
            "name_hi": "कॉपर ऑक्सीक्लोराइड 50% डब्ल्यूपी (3 ग्राम/ली) अथवा डिफेनोकोनाज़ोल 25% ईसी",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 14,
            "instructions_en": "Apply when angular brown corky necrotic pustules appear on leaves or fruit stems.",
            "instructions_hi": "पत्तियों पर भूरे रंग के उभरे हुए कैंकर धब्बे दिखने पर तुरंत छिड़काव करें।"
        },
        "safety_instructions_en": "Wait 15 days before harvesting guava fruits for fresh consumption.",
        "safety_instructions_hi": "ताजा अमरूद तोड़ने से 15 दिन पूर्व छिड़काव बंद कर दें।"
    },
    "Guava_Anthracnose": {
        "crop": "Guava",
        "disease_name_en": "Guava Anthracnose / Dieback",
        "disease_name_hi": "अमरूद का एन्थ्रेक्नोज़ / टहनी सूखा रोग (Guava Anthracnose)",
        "causal_organism": "Colletotrichum gloeosporioides (Fungus)",
        "organic_remedy": {
            "name_en": "Neem Oil 10,000 PPM + Trichoderma harzianum (4 g/L)",
            "name_hi": "नीम का तेल 10,000 पीपीएम + ट्राइकोडर्मा हरज़ियानम",
            "base_ratio_ml_per_liter": 3.5,
            "application_interval_days": 8,
            "instructions_en": "Spray on leaves showing pinhead necrotic specks. Prune and burn dried infected twigs.",
            "instructions_hi": "पत्तियों पर भूरे धब्बे दिखने पर छिड़कें। सूखी टहनियों को काटकर नष्ट करें।"
        },
        "chemical_remedy": {
            "name_en": "Mancozeb 75% WP (Dithane M-45, 2.5 g/L) or Chlorothalonil 75% WP (2 g/L)",
            "name_hi": "मैंकोजेब 75% डब्ल्यूपी (डाइथेन एम-45, 2.5 ग्राम/ली) अथवा क्लोरोथैलोनिल 75% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 12,
            "instructions_en": "Apply prophylactic spray before onset of monsoon rains and repeat at fruit set stage.",
            "instructions_hi": "मानसून से पहले बचाव हेतु छिड़काव करें और फल बनने के समय दोहराएं।"
        },
        "safety_instructions_en": "Wear eye protection and avoid skin contact. 12-day pre-harvest waiting period.",
        "safety_instructions_hi": "आंखों व त्वचा को बचाएं। फल तोड़ने से 12 दिन पहले छिड़काव न करें।"
    },
    "Lemon_Citrus_canker": {
        "crop": "Lemon (Citrus)",
        "disease_name_en": "Citrus Canker",
        "disease_name_hi": "नींबू का कैंकर रोग (Citrus Canker)",
        "causal_organism": "Xanthomonas axonopodis pv. citri (Bacterium)",
        "organic_remedy": {
            "name_en": "Neem Seed Kernel Extract (NSKE 5%) + Copper Oxychloride (2.5 g/L)",
            "name_hi": "नीम बीज अर्क (एनएसकेई 5%) + कॉपर ऑक्सीक्लोराइड",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 8,
            "instructions_en": "Control citrus leaf miner larvae which create entry wounds for canker bacteria. Prune severely infected shoots.",
            "instructions_hi": "लीफ माइनर कीट को रोकें जो कैंकर जीवाणु के प्रवेश मार्ग बनाते हैं। संक्रमित शाखाएं काटें।"
        },
        "chemical_remedy": {
            "name_en": "Streptocycline (1 g/10 L water) + Copper Oxychloride 50% WP (3 g/L)",
            "name_hi": "स्ट्रेप्टोसाइक्लिन (1 ग्राम/10 ली पानी) + कॉपर ऑक्सीक्लोराइड 50% डब्ल्यूपी",
            "base_ratio_ml_per_liter": 2.5,
            "application_interval_days": 12,
            "instructions_en": "Apply bactericidal wash when small, raised, blister-like corky lesions with yellow chlorotic halos appear on leaves.",
            "instructions_hi": "उभरे हुए खुरदरे धब्बे और पीले घेरे दिखते ही जीवाणुनाशक का संपूर्ण छिड़काव करें।"
        },
        "safety_instructions_en": "Do not spray during full bloom to safeguard pollinators. 10 days pre-harvest interval.",
        "safety_instructions_hi": "फूलों के खिलने पर छिड़काव न करें ताकि परागण सुरक्षित रहे। तुड़ाई से 10 दिन पूर्व बंद करें।"
    },
    "Lemon_Dieback": {
        "crop": "Lemon (Citrus)",
        "disease_name_en": "Lemon Twig Blight / Dieback",
        "disease_name_hi": "नींबू का डाइबैक / शीर्ण रोग (Lemon Dieback)",
        "causal_organism": "Colletotrichum gloeosporioides & Diplodia natalensis (Fungi)",
        "organic_remedy": {
            "name_en": "Pruning of dried twigs + Bordeaux Paste (10%) on cuts + Trichoderma viride",
            "name_hi": "सूखी टहनियों की छंटाई + कटे सिरों पर बोर्डो लेप + ट्राइकोडर्मा विरिडे",
            "base_ratio_ml_per_liter": 3.0,
            "application_interval_days": 14,
            "instructions_en": "Cut dried branches 2-3 inches below healthy green tissue. Seal cut ends with Bordeaux paste immediately.",
            "instructions_hi": "सूखी टहनी को हरे हिस्से से 2-3 इंच नीचे से काटें और तुरंत बोर्डो लेप लगाएं।"
        },
        "chemical_remedy": {
            "name_en": "Carbendazim 12% + Mancozeb 63% WP (Saaf, 2.0 g/L) or Copper Hydroxide (2.5 g/L)",
            "name_hi": "कार्बेन्डाजिम 12% + मैंकोजेब 63% डब्ल्यूपी (साफ, 2 ग्राम/ली) अथवा कॉपर हाइड्रॉक्साइड",
            "base_ratio_ml_per_liter": 2.0,
            "application_interval_days": 14,
            "instructions_en": "Foliar and branch drenching spray to halt wither-tip progression down the branches.",
            "instructions_hi": "टहनियों के ऊपर से नीचे की ओर सूखने को रोकने हेतु पत्तियों व शाखाओं पर छिड़काव करें।"
        },
        "safety_instructions_en": "14 days waiting period before harvesting lemons. Maintain soil drainage to prevent root hypoxia.",
        "safety_instructions_hi": "नींबू तुड़ाई से 14 दिन पहले छिड़काव रोकें। जड़ों में जलभराव न होने दें।"
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
        clean_search = disease_key.replace(",", "").replace("-", "_").replace(" ", "_").lower()
        # 1. Exact or substring match in DB keys
        for k in DISEASE_REMEDY_DATABASE.keys():
            clean_k = k.replace(",", "").replace("-", "_").replace(" ", "_").lower()
            if clean_k in clean_search or clean_search in clean_k:
                matched_key = k
                break

        # 2. Token overlap fallback
        if matched_key == "Healthy_Crop":
            words = [w for w in clean_search.split("_") if len(w) > 3 and w not in ["crop", "leaf", "plant"]]
            best_k = None
            best_match_count = 0
            for k in DISEASE_REMEDY_DATABASE.keys():
                if k == "Healthy_Crop":
                    continue
                k_lower = k.lower()
                matches = sum(1 for w in words if w in k_lower)
                if matches > best_match_count:
                    best_match_count = matches
                    best_k = k
            if best_k and best_match_count >= 1:
                matched_key = best_k

        # 3. Generic fallback for unknown diseases
        if matched_key == "Healthy_Crop" and "healthy" not in disease_key.lower():
            if "mango" in clean_search:
                matched_key = "Mango_Anthracnose"
            elif "guava" in clean_search:
                matched_key = "Guava_Canker"
            elif "lemon" in clean_search or "lime" in clean_search:
                matched_key = "Lemon_Citrus_canker"
            elif "orange" in clean_search or "citrus" in clean_search:
                matched_key = "Orange_Haunglongbing"
            elif "peach" in clean_search:
                matched_key = "Peach_Bacterial_spot"
            elif "squash" in clean_search or "cucurbit" in clean_search:
                matched_key = "Squash_Powdery_mildew"
            elif "strawberry" in clean_search:
                matched_key = "Strawberry_Leaf_scorch"
            elif "corn" in clean_search or "maize" in clean_search:
                matched_key = "Corn_Northern_Leaf_Blight"
            elif "pepper" in clean_search:
                matched_key = "Pepper_Bacterial_spot"
            else:
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
