"""
Configuration definitions for adverse event prevalence analyses.

Each dictionary in the CONFIGURATIONS list defines one complete analysis run.
The following keys are supported:

- disease (str): REQUIRED. The primary disease name. Must have a corresponding
  definition in `groups/base.csv` and a data folder in `data/cohorts/`.

- comorbidity (str): REQUIRED. The comorbidity name. Must have a corresponding
  definition in `groups/comorbidities.csv`.

- drugs (list[str]): REQUIRED. A list of one or more drug names to be grouped
  for the analysis. Each name must have a corresponding definition in `groups/drugs.csv`.

- aes (list[str]): REQUIRED. A list of one or more adverse event names to
  be analyzed. Each name must have a corresponding definition in
  `groups/adverse_effects.csv`.

- drug_group_name (str): OPTIONAL. A custom name for the drug group for cleaner
  labels in plots and outputs. If omitted, the list of drug names is used.

- enabled (bool): OPTIONAL. Set to `False` to skip this entire analysis block.
  If this key is omitted, it defaults to `True`.

- run_regression (bool): OPTIONAL. Set to `False` to disable the confounder-adjusted
  logistic regression analysis. If this key is omitted, it defaults to `True`.

- confounders (list[str]): OPTIONAL. A list of additional confounder names to include
  in the regression model. Each name must have a corresponding definition in
  `groups/confounders.csv`. The model ALWAYS adjusts for age, sex, and
  socioeconomic status by default; this list adds to that base model.
"""

CONFIGURATIONS = [
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
    # Success: BRASH
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drugs": ["atenolol", "metoprolol", "propranolol", "bisoprolol", "carvedilol", "labetalol", "timolol", "oxprenolol", "pindolol"],
        "drug_group_name": "beta-blocker",
        "aes": ["acute kidney failure", "unspecified acute kidney failure", "hyperkalemia", "cardiac dysrhythmia"],
        "enabled": False,
    },
    # Success
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drugs": ["hydrochlorothiazide", "indapamide", "furosemide", "spironolactone", "eplerenone", "amiloride", "triamterene"],
        "drug_group_name": "diuretic",
        "aes": ["squamous cell carcinoma", "non-melanoma skin cancer"],
        "negative_controls": ["corneal abrasion"],
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
    # Success
    {
        "disease": "diabetes",
        "comorbidity": "ischemic heart disease",
        "drugs": ["sitagliptin", "saxagliptin", "linagliptin", "vildagliptin"],
        "drug_group_name": "DPP-4 inhibitor",
        "aes": ["hepatocellular carcinoma"],
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
    # Success
    {
        "disease": "bronchial_asthma",
        "comorbidity": "ischemic heart disease",
        "drugs": ["salmeterol", "formoterol", "indacaterol", "olodaterol"],
        "drug_group_name": "long-acting beta-2 agonist",
        "aes": ["stroke"],
        "enabled": False,
    },
    # Success for RF, not pancreatitis
    # Pancreatitis OR success, prevalence failure
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drugs": ["metformin"],
        "aes": ["acute pancreatitis", "respiratory failure"],
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
    # Success
    {
        "disease": "hyperlipidemia",
        "comorbidity": "hypothyroidism",
        "drugs": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "fluvastatin", "cerivastatin"],
        "drug_group_name": "statin",
        "aes": ["liver failure"],
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
    # Positive control: success
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drugs": ["captopril", "enalapril", "ramipril", "benazepril", "fosinopril", "cilazapril"],
        "drug_group_name": "ACE inhibitor",
        "aes": ["acute kidney failure", "hyperkalemia"],
        "enabled": False,
    },
    # Positive control: weak success
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drugs": ["metformin"],
        "aes": ["acidosis"],
        "enabled": False,
    },
    # Positive control: weak success
    {
        "disease": "hyperlipidemia",
        "comorbidity": "hypothyroidism",
        "drugs": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "fluvastatin", "cerivastatin"],
        "drug_group_name": "statin",
        "aes": ["rhabdomyolysis"],
        "enabled": False,
    },
    # Positive control: success
    {
        "disease": "bronchial_asthma",
        "comorbidity": "ischemic heart disease",
        "drugs": ["salbutamol", "terbutaline", "salmeterol", "formoterol", "indacaterol", "olodaterol"],
        "drug_group_name": "beta-2 agonist",
        "aes": ["angina"],
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
    # Negative control: succcess
    # https://chatgpt.com/share/691359d5-f740-8012-ae44-4de58a433dd3
    # https://www.acc.org/Latest-in-Cardiology/Clinical-Trials/2015/09/17/10/11/EMPA-REG-OUTCOME#:~:text=Secondary%20outcomes%3A
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drugs": ["empagliflozin", "dapagliflozin", "ertugliflozin"],
        "drug_group_name": "SGLT-2 inhibitor",
        "aes": ["acute kidney failure", "heart failure"],
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
    # Negative control: success
    # https://chatgpt.com/c/6911e84a-9d68-8328-8b07-03a3f3423bd2
    # https://www.acc.org/Latest-in-Cardiology/Clinical-Trials/2016/06/13/14/24/LEADER
    {
        "disease": "diabetes",
        "comorbidity": "ischemic heart disease",
        "drugs": ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide"],
        "drug_group_name": "GLP-1 receptor agonist",
        "aes": ["stroke", "myocardial infarction", "heart failure"],
        "enabled": True,
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
