
DEPRECATED_CONFIGURATIONS = [
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drugs": ["dulaglutide"],
        "aes": ["pancreatitis", "hypoglycemia"],
        "enabled": False,
    },
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drugs": ["losartan", "valsartan", "irbesartan", "candesartan", "olmesartan"],
        "drug_group_name": "ARB",
        "aes": ["hyperkalemia", "systemic lupus erythematosus"],
        "enabled": False,
    },
    # Failure
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drugs": ["valsartan", "irbesartan", "candesartan", "olmesartan"],
        "drug_group_name": "non-losartan ARB",
        "aes": ["systemic lupus erythematosus"],
        "enabled": False,
    },
    # Failure: tinidazole is used to treat bacterial vaginosis
    {
        "disease": "vaginosis",
        "comorbidity": "pregnancy",
        "drugs": ["tinidazole"],
        "aes": ["neutropenia"],
        "enabled": False,
    },
    # Failure
    {
        "disease": "diabetes",
        "comorbidity": "surgical follow up",
        "drugs": ["empagliflozin", "dapagliflozin", "ertugliflozin"],
        "drug_group_name": "SGLT-2 inhibitor",
        "aes": ["muscle weakness"],
        "enabled": False,
    },
    # Failure (OR success, prevalence failure)
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drugs": ["captopril", "enalapril", "ramipril", "benazepril", "fosinopril", "cilazapril"],
        "drug_group_name": "ACE inhibitor",
        # "aes": ["taste and smell alteration", "muscle weakness"],
        "aes": ["pancreatitis", "ataxia"],
        "enabled": False,
    },
    # Positive control: failure
    {
        "disease": "diabetes",
        "comorbidity": "surgical follow up",
        "drugs": ["empagliflozin", "dapagliflozin", "ertugliflozin"],
        "drug_group_name": "SGLT-2 inhibitor",
        "aes": ["urinary tract infection"],
        "enabled": False,
    },
    # Negative control: failure
    {
        "disease": "bronchial_asthma",
        "comorbidity": "ischemic heart disease",
        "drugs": ["atenolol", "metoprolol", "bisoprolol"],
        "drug_group_name": "cardioselective beta-blocker",
        "aes": ["respiratory failure"],
        "enabled": False,
    },
    # Negative control: failure (prevalence success, OR failure)
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drugs": ["captopril", "enalapril", "ramipril", "benazepril", "fosinopril", "cilazapril"],
        "drug_group_name": "ACE inhibitor",
        "aes": ["influenza"],
        "enabled": False,
    },
    # Negative control: failure (confounded by disease severity)
    # https://chatgpt.com/share/691359d5-f740-8012-ae44-4de58a433dd3
    {
        "disease": "hyperlipidemia",
        "comorbidity": "ischemic heart disease",
        "drugs": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "fluvastatin", "cerivastatin"],
        "drug_group_name": "statin",
        "aes": ["stroke", "myocardial infarction"],
        "enabled": False,
    },
    # Negative control: failure (confounded by disease severity)
    # https://chatgpt.com/share/691359d5-f740-8012-ae44-4de58a433dd3
    {
        "disease": "hypertension",
        "comorbidity": "ischemic heart disease",
        "drugs": ["atenolol", "metoprolol", "propranolol", "bisoprolol", "carvedilol", "labetalol", "timolol", "oxprenolol", "pindolol"],
        "drug_group_name": "beta-blocker",
        "aes": ["heart failure", "myocardial infarction"],
        "enabled": False,
    },
    # Negative control: failure
    # https://chatgpt.com/share/691359d5-f740-8012-ae44-4de58a433dd3
    {
        "disease": "diabetes",
        "comorbidity": "ischemic heart disease",
        "drugs": ["captopril", "enalapril", "ramipril", "benazepril", "fosinopril", "cilazapril"],
        "drug_group_name": "ACE inhibitor",
        "aes": ["stroke", "myocardial infarction"],
        "enabled": False,
    },
    # Negative control: failure
    {
        "disease": "bipolar_disorder",
        "comorbidity": "hypothyroidism",
        "drugs": ["lithium"],
        "aes": ["suicide attempt"],
        "enabled": False,
    },
    # Negative control: failure
    # https://chatgpt.com/c/6912a3c7-e284-8331-b7de-2b01e313921e
    {
        "disease": "hypertension",
        "comorbidity": "ischemic heart disease",
        "drugs": ["atenolol", "metoprolol", "propranolol", "bisoprolol", "carvedilol", "labetalol", "timolol", "oxprenolol", "pindolol"],
        "drug_group_name": "beta-blocker",
        "aes": ["migraine", "essential tremor"],
        "enabled": False,
    },
    # Negative control: failure
    # https://chatgpt.com/c/6912a3c7-e284-8331-b7de-2b01e313921e
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drugs": ["captopril", "enalapril", "ramipril", "benazepril", "fosinopril", "cilazapril"],
        "drug_group_name": "ACE inhibitor",
        "aes": ["migraine"],
        "enabled": False,
    },
    # Negative control: failure
    # https://chatgpt.com/c/6912a3c7-e284-8331-b7de-2b01e313921e
    {
        "disease": "hyperlipidemia",
        "comorbidity": "ischemic heart disease",
        "drugs": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "fluvastatin", "cerivastatin"],
        "drug_group_name": "statin",
        "aes": ["cholelithiasis"],
        "enabled": False,
    },

    # Negative control, unrelated phenotype: 
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drugs": ["hydrochlorothiazide", "indapamide", "furosemide", "spironolactone", "eplerenone", "amiloride", "triamterene"],
        "drug_group_name": "diuretic",
        "aes": ["migraine"],
        "enabled": False,
    },
]