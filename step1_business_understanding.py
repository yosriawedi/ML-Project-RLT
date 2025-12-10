"""
CRISP-DM STEP 1: BUSINESS UNDERSTANDING
Comprehensive Dataset Analysis and Problem Identification
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Define workspace path
workspace_path = r'c:\Users\DELL\Downloads\(No subject)'

print("=" * 100)
print("CRISP-DM STEP 1: BUSINESS UNDERSTANDING")
print("=" * 100)
print("\n")

datasets_info = []

# ==================== DATASET 1: BOSTON HOUSING ====================
print("\n" + "=" * 100)
print("DATASET 1: BOSTON HOUSING")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'BostonHousing.csv'))
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    datasets_info.append({
        'name': 'BostonHousing',
        'file': 'BostonHousing.csv',
        'problem_type': 'REGRESSION',
        'target': 'medv' if 'medv' in df.columns else df.columns[-1],
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'Predict median home values in Boston suburbs',
        'domain': 'Real Estate',
        'goal': 'Predict housing prices based on neighborhood characteristics',
        'risks': 'Outdated data, potential ethical concerns with certain features',
        'suitable_models': 'Linear Regression, Random Forest, Gradient Boosting, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 2: WINE QUALITY RED ====================
print("\n" + "=" * 100)
print("DATASET 2: WINE QUALITY (RED)")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'winequality-red.csv'), sep=';')
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    target_col = 'quality'
    is_classification = df[target_col].nunique() <= 10 if target_col in df.columns else True
    
    datasets_info.append({
        'name': 'WineQuality_Red',
        'file': 'winequality-red.csv',
        'problem_type': 'CLASSIFICATION' if is_classification else 'REGRESSION',
        'target': target_col,
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'Predict wine quality based on physicochemical tests',
        'domain': 'Food & Beverage',
        'goal': 'Classify wine quality (3-8 scale) from chemical properties',
        'risks': 'Imbalanced classes, subjective quality ratings',
        'suitable_models': 'Logistic Regression, Random Forest, XGBoost, RLT with variable muting'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 3: WINE QUALITY WHITE ====================
print("\n" + "=" * 100)
print("DATASET 3: WINE QUALITY (WHITE)")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'winequality-white.csv'), sep=';')
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    target_col = 'quality'
    is_classification = df[target_col].nunique() <= 10 if target_col in df.columns else True
    
    datasets_info.append({
        'name': 'WineQuality_White',
        'file': 'winequality-white.csv',
        'problem_type': 'CLASSIFICATION' if is_classification else 'REGRESSION',
        'target': target_col,
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'Predict white wine quality based on physicochemical tests',
        'domain': 'Food & Beverage',
        'goal': 'Classify wine quality from chemical properties',
        'risks': 'Larger dataset than red wine, may have different patterns',
        'suitable_models': 'Random Forest, Gradient Boosting, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 4: SONAR ====================
print("\n" + "=" * 100)
print("DATASET 4: SONAR")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'sonar data.csv'), header=None)
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Target values: {df.iloc[:, -1].unique()}")
    
    datasets_info.append({
        'name': 'Sonar',
        'file': 'sonar data.csv',
        'problem_type': 'BINARY CLASSIFICATION',
        'target': 'Class (R=Rock, M=Mine)',
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'Classify sonar signals bounced off metal cylinder vs rocks',
        'domain': 'Signal Processing / Defense',
        'goal': 'Distinguish between sonar returns from mines vs rocks',
        'risks': 'High-dimensional (60 features), potential noise variables',
        'suitable_models': 'SVM, Random Forest, RLT with variable muting (ideal for high-dim)'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 5: PARKINSONS ====================
print("\n" + "=" * 100)
print("DATASET 5: PARKINSON'S DISEASE")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'parkinsons.data'))
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    datasets_info.append({
        'name': 'Parkinsons',
        'file': 'parkinsons.data',
        'problem_type': 'BINARY CLASSIFICATION',
        'target': 'status',
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 2,  # Excluding name and status
        'description': 'Predict Parkinson\'s disease from voice measurements',
        'domain': 'Healthcare / Medical Diagnosis',
        'goal': 'Detect Parkinson\'s disease from voice recording features',
        'risks': 'Medical application - requires high accuracy, imbalanced data',
        'suitable_models': 'Logistic Regression, Random Forest, XGBoost, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 6: BREAST CANCER (WDBC) ====================
print("\n" + "=" * 100)
print("DATASET 6: WISCONSIN BREAST CANCER DIAGNOSTIC")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'wdbc.data'), header=None)
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Target values: {df.iloc[:, 1].unique()}")
    
    datasets_info.append({
        'name': 'WDBC_BreastCancer',
        'file': 'wdbc.data',
        'problem_type': 'BINARY CLASSIFICATION',
        'target': 'Diagnosis (M=Malignant, B=Benign)',
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 2,  # Excluding ID and diagnosis
        'description': 'Predict breast cancer diagnosis from cell nucleus measurements',
        'domain': 'Healthcare / Oncology',
        'goal': 'Classify tumors as malignant or benign',
        'risks': 'Critical medical application, false negatives are costly',
        'suitable_models': 'Logistic Regression, SVM, Random Forest, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 7: CONCRETE STRENGTH ====================
print("\n" + "=" * 100)
print("DATASET 7: CONCRETE COMPRESSIVE STRENGTH")
print("=" * 100)
try:
    df = pd.read_excel(os.path.join(workspace_path, 'Concrete_Data.xls'))
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    datasets_info.append({
        'name': 'Concrete',
        'file': 'Concrete_Data.xls',
        'problem_type': 'REGRESSION',
        'target': 'Concrete compressive strength',
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'Predict concrete strength from mixture components',
        'domain': 'Civil Engineering / Materials Science',
        'goal': 'Predict compressive strength based on concrete ingredients',
        'risks': 'Non-linear relationships, interaction effects between ingredients',
        'suitable_models': 'Random Forest, Gradient Boosting, Neural Networks, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 8: AUTO MPG ====================
print("\n" + "=" * 100)
print("DATASET 8: AUTO MPG")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'auto-mpg.data'), 
                     delim_whitespace=True, header=None,
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 
                            'weight', 'acceleration', 'model_year', 'origin', 'car_name'])
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    datasets_info.append({
        'name': 'AutoMPG',
        'file': 'auto-mpg.data',
        'problem_type': 'REGRESSION',
        'target': 'mpg',
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 2,  # Excluding mpg and car_name
        'description': 'Predict fuel efficiency (MPG) from car attributes',
        'domain': 'Automotive / Energy',
        'goal': 'Predict miles per gallon based on car specifications',
        'risks': 'Missing values in horsepower, categorical features need encoding',
        'suitable_models': 'Linear Regression, Random Forest, Gradient Boosting, RLT'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== DATASET 9: SCHOOL DATA ====================
print("\n" + "=" * 100)
print("DATASET 9: SCHOOL DATA")
print("=" * 100)
try:
    df = pd.read_csv(os.path.join(workspace_path, 'data_school.csv'))
    print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ First few rows:")
    print(df.head())
    
    # Try to infer problem type
    last_col = df.columns[-1]
    is_classification = df[last_col].nunique() <= 20
    
    datasets_info.append({
        'name': 'SchoolData',
        'file': 'data_school.csv',
        'problem_type': 'CLASSIFICATION' if is_classification else 'REGRESSION',
        'target': last_col,
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'description': 'School-related prediction task (to be determined)',
        'domain': 'Education',
        'goal': 'TBD - Will be determined during data exploration',
        'risks': 'Unknown data structure',
        'suitable_models': 'To be determined'
    })
except Exception as e:
    print(f"✗ Error loading: {e}")

# ==================== SUMMARY ====================
print("\n" + "=" * 100)
print("BUSINESS UNDERSTANDING SUMMARY")
print("=" * 100)
print(f"\nTotal datasets identified: {len(datasets_info)}")
print("\n")

# Create summary DataFrame
summary_df = pd.DataFrame(datasets_info)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(os.path.join(workspace_path, 'datasets_summary.csv'), index=False)
print("\n✓ Summary saved to: datasets_summary.csv")

# ==================== PROBLEM TYPE ANALYSIS ====================
print("\n" + "=" * 100)
print("PROBLEM TYPE DISTRIBUTION")
print("=" * 100)
problem_counts = summary_df['problem_type'].value_counts()
for ptype, count in problem_counts.items():
    print(f"• {ptype}: {count} dataset(s)")

# ==================== RLT APPLICABILITY ANALYSIS ====================
print("\n" + "=" * 100)
print("RLT METHODOLOGY APPLICABILITY")
print("=" * 100)
print("""
Based on Zhu2015 paper, RLT is especially effective for:
1. ✓ High-dimensional datasets (many features)
2. ✓ Sparse signal structure (few strong variables among many noise variables)
3. ✓ Tree-based models that benefit from variable muting
4. ✓ Problems where feature selection is critical

IDEAL CANDIDATES FOR RLT:
""")

for idx, info in enumerate(datasets_info, 1):
    rlt_score = 0
    reasons = []
    
    # High dimensionality
    if info['n_features'] >= 20:
        rlt_score += 2
        reasons.append("High-dimensional")
    elif info['n_features'] >= 10:
        rlt_score += 1
        reasons.append("Medium-dimensional")
    
    # Classification problems benefit from variable muting
    if 'CLASSIFICATION' in info['problem_type']:
        rlt_score += 1
        reasons.append("Classification task")
    
    # Medical/signal processing domains often have noise
    if info['domain'] in ['Healthcare / Medical Diagnosis', 'Signal Processing / Defense', 
                          'Healthcare / Oncology']:
        rlt_score += 1
        reasons.append("High-noise domain")
    
    priority = "⭐⭐⭐ HIGH" if rlt_score >= 3 else "⭐⭐ MEDIUM" if rlt_score >= 2 else "⭐ LOW"
    
    print(f"\n{idx}. {info['name']} ({info['problem_type']})")
    print(f"   Priority: {priority}")
    print(f"   Features: {info['n_features']}")
    print(f"   Samples: {info['n_samples']}")
    print(f"   Reasons: {', '.join(reasons)}")

print("\n" + "=" * 100)
print("STEP 1 COMPLETE ✓")
print("=" * 100)
print("\nNext: STEP 2 - DATA UNDERSTANDING")
