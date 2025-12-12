# -*- coding: utf-8 -*-
"""
🌲 RLT Extra Trees: Étude Comparative Complète
Reinforcement Learning Trees - Analyse Multi-Modèles

Author: Dhia Romdhane
Date: December 2025
Méthodologie: CRISP-DM

INSTRUCTIONS:
1. Ouvrez Google Colab
2. Créez un nouveau notebook  
3. Copiez-collez ce code dans plusieurs cellules
4. Exécutez cellule par cellule

OBJECTIF:
Comparer RLT-ExtraTrees contre 7 autres modèles:
1. RLT-ExtraTrees (Reinforcement Learning Trees)
2. RF (Random Forest classique)
3. RF-√p (Random Forest avec mtry = √p)
4. RF-log(p) (Random Forest avec mtry = log(p))
5. ExtraTrees (Extra Trees standard)
6. BART (Bayesian Additive Regression Trees) 
7. LASSO (Régression LASSO)
8. Boosting (XGBoost)
"""

# ==============================================================================
# CELLULE 1: INSTALLATION DES BIBLIOTHÈQUES
# ==============================================================================

# Installation
!pip install xgboost scikit-learn pandas numpy matplotlib seaborn scipy -q

print("✅ Bibliothèques installées!")

# ==============================================================================
# CELLULE 2: IMPORTATION DES BIBLIOTHÈQUES
# ==============================================================================

# Data manipulation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Statistical tests
from scipy.stats import f_oneway, pearsonr

# File handling
from google.colab import files
import io
import time

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ Bibliothèques importées!")
print(f"📌 Random State: {RANDOM_STATE}")

# ==============================================================================
# CELLULE 3: CONFIGURATION DES HYPERPARAMÈTRES (TOUS FIXES!)
# ==============================================================================

print("="*70)
print("⚙️  CONFIGURATION DES HYPERPARAMÈTRES - FIXES POUR TOUS LES MODÈLES")
print("="*70)

# General settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1

# RLT Configuration
VI_THRESHOLD = 0.01
VI_EXTRA_TREES_WEIGHT = 0.5
VI_STAT_WEIGHT = 0.5

# Tree-based models configuration (FIXED!)
TREE_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}

# XGBoost configuration (FIXED!)
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
    'verbosity': 0
}

# LASSO configuration (FIXED!)
LASSO_CONFIG = {
    'alpha': 0.1,
    'random_state': RANDOM_STATE,
    'max_iter': 1000
}

# Boosting configuration (FIXED!)
BOOSTING_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

print(f"\\n📊 Paramètres Généraux:")
print(f"   - Random State: {RANDOM_STATE}")
print(f"   - Test Size: {TEST_SIZE}")
print(f"   - CV Folds: {CV_FOLDS}")
print(f"\\n🌲 Paramètres Tree-Based:")
print(f"   - n_estimators: {TREE_CONFIG['n_estimators']}")
print(f"   - max_depth: {TREE_CONFIG['max_depth']}")
print(f"\\n🔍 Paramètres RLT:")
print(f"   - VI Threshold: {VI_THRESHOLD}")
print(f"   - VI Extra Trees Weight: {VI_EXTRA_TREES_WEIGHT}")
print(f"   - VI Statistical Weight: {VI_STAT_WEIGHT}")
print("\\n✅ Configuration chargée!")

# ==============================================================================
# CELLULE 4: UPLOAD DE DATASET
# ==============================================================================

print("="*70)
print("📁 UPLOAD DE DATASET")
print("="*70)
print("\\n👉 Sélectionnez votre fichier CSV")
print("   Format: CSV avec header, dernière colonne = target\\n")

uploaded = files.upload()

# Get filename
filename = list(uploaded.keys())[0]
print(f"\\n✅ Fichier uploadé: {filename}")

# Load dataset
df = pd.read_csv(io.BytesIO(uploaded[filename]))

print(f"\\n📊 Dataset chargé:")
print(f"   - Shape: {df.shape}")
print(f"   - Samples: {df.shape[0]}")
print(f"   - Features: {df.shape[1] - 1}")
print(f"   - Target: {df.columns[-1]}")

print("\\n📋 Aperçu des données:")
display(df.head())

print("\\n📈 Informations:")
df.info()

# ==============================================================================
# CELLULE 5: CRISP-DM - DATA UNDERSTANDING (EDA)
# ==============================================================================

print("="*70)
print("📊 CRISP-DM: DATA UNDERSTANDING")
print("="*70)

# Separate features and target
target_col = df.columns[-1]
features = df.columns[:-1]

print(f"\\n🎯 Target: {target_col}")
print(f"📊 Features ({len(features)}): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

# Statistics
print("\\n📈 Statistiques des Features:")
display(df[features].describe().T)

# Target statistics
print(f"\\n🎯 Statistiques du Target '{target_col}':")
if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
    print("   Type: Classification")
    print(f"\\n   Distribution:")
    print(df[target_col].value_counts())
    print(f"\\n   Proportions:")
    print(df[target_col].value_counts(normalize=True))
else:
    print("   Type: Régression")
    print(f"\\n   Statistiques:")
    print(df[target_col].describe())

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

print("\\n❓ Valeurs Manquantes:")
if len(missing_df) > 0:
    print(f"   ⚠️  {len(missing_df)} colonnes avec valeurs manquantes")
    display(missing_df)
else:
    print("   ✅ Aucune valeur manquante")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\\n🔄 Doublons:")
if duplicates > 0:
    print(f"   ⚠️  {duplicates} lignes dupliquées ({duplicates/len(df)*100:.2f}%)")
else:
    print(f"   ✅ Aucun doublon")

# Visualizations
print("\\n📊 Génération des visualisations...")

# Target distribution
plt.figure(figsize=(12, 5))

if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
    plt.subplot(1, 2, 1)
    df[target_col].value_counts().plot(kind='bar', color='steelblue')
    plt.title(f'Distribution de {target_col}', fontsize=14, fontweight='bold')
    plt.xlabel(target_col)
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    plt.title(f'Proportions de {target_col}', fontsize=14, fontweight='bold')
    plt.ylabel('')
else:
    plt.subplot(1, 2, 1)
    plt.hist(df[target_col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution de {target_col}', fontsize=14, fontweight='bold')
    plt.xlabel(target_col)
    plt.ylabel('Fréquence')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df[target_col], vert=True)
    plt.title(f'Box Plot de {target_col}', fontsize=14, fontweight='bold')
    plt.ylabel(target_col)

plt.tight_layout()
plt.show()

# Correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=len(numeric_df.columns) <= 15, fmt=".2f", 
                cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    if target_col in numeric_df.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        print(f"\\n🎯 Top 10 Corrélations avec {target_col}:")
        print(target_corr.head(10))

print("\\n✅ Data Understanding terminé!")

# ==============================================================================
# CELLULE 6: CRISP-DM - DATA PREPARATION
# ==============================================================================

print("="*70)
print("🔧 CRISP-DM: DATA PREPARATION")
print("="*70)

# 1. Detect problem type
target_col = df.columns[-1]
unique_values = df[target_col].nunique()

if df[target_col].dtype == 'object' or unique_values < 10:
    problem_type = 'classification'
    print(f"\\n✅ Type: CLASSIFICATION")
    print(f"   - Target: {target_col}")
    print(f"   - Classes: {unique_values}")
else:
    problem_type = 'regression'
    print(f"\\n✅ Type: RÉGRESSION")
    print(f"   - Target: {target_col}")
    print(f"   - Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")

# 2. Clean data
df_clean = df.copy()
initial_shape = df_clean.shape

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"\\n🧹 Doublons supprimés: {initial_shape[0] - df_clean.shape[0]}")

# Handle missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

# 3. Separate features and target
X = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]

# 4. Encode categorical features
categorical_features = X.select_dtypes(include=['object']).columns
if len(categorical_features) > 0:
    print(f"\\n🔄 Encoding {len(categorical_features)} categorical features...")
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
else:
    X_encoded = X.copy()

# 5. Encode target if classification
if problem_type == 'classification':
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"\\n🎯 Target encodé: {le.classes_}")
    else:
        y_encoded = y.values
else:
    y_encoded = y.values

# 6. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

print(f"\\n📏 Features scaled (StandardScaler)")
print(f"   - Shape: {X_scaled.shape}")
print(f"   - Features: {X_scaled.shape[1]}")

# 7. Split train/test
if problem_type == 'classification':
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

print(f"\\n✂️  Split Train/Test:")
print(f"   - Train: {X_train.shape[0]} samples ({(1-TEST_SIZE)*100:.0f}%)")
print(f"   - Test: {X_test.shape[0]} samples ({TEST_SIZE*100:.0f}%)")
print(f"   - Features: {X_train.shape[1]}")

print("\\n✅ Data Preparation terminé!")
print("🎯 Données prêtes pour la modélisation!")

# ==============================================================================
# CELLULE 7: RLT - VARIABLE IMPORTANCE
# ==============================================================================

print("="*70)
print("🧠 RLT: VARIABLE IMPORTANCE")
print("="*70)

def compute_rlt_variable_importance(X, y, problem_type):
    """
    Compute Variable Importance using Extra Trees + Statistical tests
    """
    print("\\n📊 Calcul de Variable Importance...")
    
    # 1. Extra Trees VI
    if problem_type == 'classification':
        et = ExtraTreesClassifier(**TREE_CONFIG)
    else:
        et = ExtraTreesRegressor(**TREE_CONFIG)
    
    et.fit(X, y)
    vi_et = et.feature_importances_
    print(f"   ✅ Extra Trees VI calculé")
    
    # 2. Statistical VI
    vi_stat = np.zeros(X.shape[1])
    for i, col in enumerate(X.columns):
        try:
            if problem_type == 'classification':
                groups = [X[col][y == label] for label in np.unique(y)]
                f_stat, _ = f_oneway(*groups)
                vi_stat[i] = f_stat / 1000.0
            else:
                corr, _ = pearsonr(X[col], y)
                vi_stat[i] = abs(corr)
        except:
            vi_stat[i] = 0
    
    print(f"   ✅ Statistical VI calculé")
    
    # 3. Normalize
    vi_et = vi_et / vi_et.sum() if vi_et.sum() > 0 else vi_et
    vi_stat = vi_stat / vi_stat.sum() if vi_stat.sum() > 0 else vi_stat
    
    # 4. Aggregate
    vi_aggregate = VI_EXTRA_TREES_WEIGHT * vi_et + VI_STAT_WEIGHT * vi_stat
    
    # 5. Create DataFrame
    vi_df = pd.DataFrame({
        'Feature': X.columns,
        'VI_ExtraTrees': vi_et,
        'VI_Statistical': vi_stat,
        'VI_Aggregate': vi_aggregate
    }).sort_values('VI_Aggregate', ascending=False)
    
    return vi_df

# Compute VI
vi_scores = compute_rlt_variable_importance(X_train, y_train, problem_type)

print(f"\\n🔝 Top 15 Features par VI:")
display(vi_scores.head(15))

# Plot VI
plt.figure(figsize=(12, 6))
top_features = vi_scores.head(20)
plt.barh(range(len(top_features)), top_features['VI_Aggregate'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Variable Importance')
plt.title('Top 20 Features par Variable Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Apply muting with ADAPTIVE threshold
print(f"\n🔇 Variable Muting (Seuil Adaptatif):")

# Calculate adaptive threshold based on VI distribution
vi_values = vi_scores['VI_Aggregate'].values
vi_median = np.median(vi_values)
vi_mean = np.mean(vi_values)
vi_std = np.std(vi_values)

# Adaptive threshold: use mean of VI or percentile-based
# Strategy: Keep top features above mean or use percentile
adaptive_threshold = max(VI_THRESHOLD, vi_mean)  # At least VI_THRESHOLD, but adaptive to data

# Alternative: percentile-based (keep top 60% of features)
# percentile_threshold = np.percentile(vi_values, 40)
# adaptive_threshold = max(VI_THRESHOLD, percentile_threshold)

print(f"   📊 Statistiques VI:")
print(f"      - Médiane: {vi_median:.4f}")
print(f"      - Moyenne: {vi_mean:.4f}")
print(f"      - Écart-type: {vi_std:.4f}")
print(f"   🎯 Seuil fixe (config): {VI_THRESHOLD}")
print(f"   ⚡ Seuil adaptatif (utilisé): {adaptive_threshold:.4f}")

high_vi_features = vi_scores[vi_scores['VI_Aggregate'] >= adaptive_threshold]['Feature'].tolist()
low_vi_features = vi_scores[vi_scores['VI_Aggregate'] < adaptive_threshold]['Feature'].tolist()

# Ensure minimum features
if len(high_vi_features) < 5:
    print(f"   ⚠️  Seuil trop strict, gardons au moins 5 features")
    high_vi_features = vi_scores.head(5)['Feature'].tolist()
    low_vi_features = vi_scores.iloc[5:]['Feature'].tolist()
    adaptive_threshold = vi_scores.iloc[4]['VI_Aggregate']

# Create muted datasets
X_train_muted = X_train[high_vi_features]
X_test_muted = X_test[high_vi_features]

print(f"\n   ✂️  Résultat du Muting:")
print(f"   - Original features: {X_train.shape[1]}")
print(f"   - Features mutées: {len(low_vi_features)} ({len(low_vi_features)/X_train.shape[1]*100:.1f}%)")
print(f"   - Features gardées: {len(high_vi_features)} ({len(high_vi_features)/X_train.shape[1]*100:.1f}%)")
print(f"   - Seuil final: {adaptive_threshold:.4f}")

print("\n✅ Variable Importance terminé!")

# ==============================================================================
# CELLULE 7b: RLT - LINEAR COMBINATIONS
# ==============================================================================

print("\n" + "="*70)
print("🔗 RLT: COMBINAISONS LINÉAIRES")
print("="*70)

def create_linear_combinations(X, vi_scores, top_n=10):
    """
    Create linear combinations of top features
    Based on RLT methodology (Zhu et al. 2015)
    """
    print("\n🔢 Création de combinaisons linéaires...")
    
    X_combined = X.copy()
    top_features = vi_scores.head(top_n)['Feature'].tolist()
    
    # Ensure we have at least 2 features
    if len(top_features) < 2:
        print("   ⚠️  Pas assez de features pour combinaisons")
        return X_combined
    
    combinations_created = 0
    
    # Create combinations between top features
    for i in range(min(5, len(top_features)-1)):
        for j in range(i+1, min(i+3, len(top_features))):
            feat1 = top_features[i]
            feat2 = top_features[j]
            
            # Weighted combination based on VI
            w1 = vi_scores[vi_scores['Feature'] == feat1]['VI_Aggregate'].values[0]
            w2 = vi_scores[vi_scores['Feature'] == feat2]['VI_Aggregate'].values[0]
            
            # Normalize weights
            total_w = w1 + w2
            w1_norm = w1 / total_w if total_w > 0 else 0.5
            w2_norm = w2 / total_w if total_w > 0 else 0.5
            
            # Create linear combination
            new_col_name = f"LC_{i}_{j}"
            X_combined[new_col_name] = w1_norm * X[feat1] + w2_norm * X[feat2]
            combinations_created += 1
    
    print(f"   ✅ {combinations_created} combinaisons linéaires créées")
    print(f"   📊 Features totales: {X.shape[1]} → {X_combined.shape[1]}")
    
    return X_combined

# Create linear combinations for RLT
# IMPORTANT: Only use VI scores for features that survived muting!
vi_scores_muted = vi_scores[vi_scores['Feature'].isin(X_train_muted.columns)].copy()
vi_scores_muted = vi_scores_muted.reset_index(drop=True)

print(f"\n📋 Features disponibles pour combinaisons: {len(vi_scores_muted)}")

X_train_rlt = create_linear_combinations(X_train_muted, vi_scores_muted)
X_test_rlt = create_linear_combinations(X_test_muted, vi_scores_muted)

print(f"\n📊 Dataset RLT final:")
print(f"   - Features originales (après muting): {X_train_muted.shape[1]}")
print(f"   - Features avec combinaisons: {X_train_rlt.shape[1]}")
print(f"   - Combinaisons ajoutées: {X_train_rlt.shape[1] - X_train_muted.shape[1]}")

print("\n✅ Combinaisons linéaires créées!")

# ==============================================================================
# CELLULE 8: MODÉLISATION - DÉFINITION DES MODÈLES
# ==============================================================================

print("="*70)
print("🤖 MODÉLISATION: DÉFINITION DES 8 MODÈLES")
print("="*70)

# Calculate mtry values
p = X_train.shape[1]
mtry_sqrt = max(1, int(np.sqrt(p)))
mtry_log = max(1, int(np.log(p)))

print(f"\n📊 Paramètres mtry:")
print(f"   - p (total features): {p}")
print(f"   - √p: {mtry_sqrt}")
print(f"   - log(p): {mtry_log:.2f} → {mtry_log}")

# Define models
models = {}

if problem_type == 'classification':
    models = {
        '1. RLT-ExtraTrees': {
            'model': ExtraTreesClassifier(**TREE_CONFIG),
            'X_train': X_train_rlt,
            'X_test': X_test_rlt,
            'description': 'RLT avec VI + Muting + Linear Combinations'
        },
        '2. RF': {
            'model': RandomForestClassifier(**TREE_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Random Forest classique'
        },
        '3. RF-√p': {
            'model': RandomForestClassifier(**{**TREE_CONFIG, 'max_features': mtry_sqrt}),
            'X_train': X_train,
            'X_test': X_test,
            'description': f'Random Forest avec mtry = √p = {mtry_sqrt}'
        },
        '4. RF-log(p)': {
            'model': RandomForestClassifier(**{**TREE_CONFIG, 'max_features': mtry_log}),
            'X_train': X_train,
            'X_test': X_test,
            'description': f'Random Forest avec mtry = log(p) = {mtry_log}'
        },
        '5. ExtraTrees': {
            'model': ExtraTreesClassifier(**TREE_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Extra Trees standard'
        },
        '6. LASSO': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', C=1/LASSO_CONFIG['alpha'], 
                                       random_state=RANDOM_STATE, max_iter=LASSO_CONFIG['max_iter']),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Logistic Regression avec LASSO'
        },
        '7. XGBoost': {
            'model': XGBClassifier(**XGBOOST_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'XGBoost Classifier'
        },
        '8. AdaBoost': {
            'model': AdaBoostClassifier(**BOOSTING_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'AdaBoost Classifier'
        }
    }
else:  # Regression
    models = {
        '1. RLT-ExtraTrees': {
            'model': ExtraTreesRegressor(**TREE_CONFIG),
            'X_train': X_train_rlt,
            'X_test': X_test_rlt,
            'description': 'RLT avec VI + Muting + Linear Combinations'
        },
        '2. RF': {
            'model': RandomForestRegressor(**TREE_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Random Forest classique'
        },
        '3. RF-√p': {
            'model': RandomForestRegressor(**{**TREE_CONFIG, 'max_features': mtry_sqrt}),
            'X_train': X_train,
            'X_test': X_test,
            'description': f'Random Forest avec mtry = √p = {mtry_sqrt}'
        },
        '4. RF-log(p)': {
            'model': RandomForestRegressor(**{**TREE_CONFIG, 'max_features': mtry_log}),
            'X_train': X_train,
            'X_test': X_test,
            'description': f'Random Forest avec mtry = log(p) = {mtry_log}'
        },
        '5. ExtraTrees': {
            'model': ExtraTreesRegressor(**TREE_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Extra Trees standard'
        },
        '6. LASSO': {
            'model': Lasso(**LASSO_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'LASSO Regression'
        },
        '7. XGBoost': {
            'model': XGBRegressor(**XGBOOST_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'XGBoost Regressor'
        },
        '8. GradientBoosting': {
            'model': GradientBoostingRegressor(**BOOSTING_CONFIG),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Gradient Boosting Regressor'
        }
    }

print(f"\n📋 Modèles définis:")
for name, config in models.items():
    print(f"   {name}: {config['description']}")
    print(f"      → Features: {config['X_train'].shape[1]}")

print("\n✅ Modèles configurés!")

# ==============================================================================
# CELLULE 9: ENTRAÎNEMENT DES MODÈLES
# ==============================================================================

print("="*70)
print("🚀 ENTRAÎNEMENT DES 8 MODÈLES")
print("="*70)

results = []

for model_name, config in models.items():
    print(f"\n{'='*60}")
    print(f"🏃 Entraînement: {model_name}")
    print(f"{'='*60}")
    
    model = config['model']
    X_tr = config['X_train']
    X_te = config['X_test']
    
    print(f"   📊 Features utilisées: {X_tr.shape[1]}")
    print(f"   ⏳ Entraînement en cours...")
    
    start_time = time.time()
    
    # Train
    model.fit(X_tr, y_train)
    
    # Predict
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    
    train_time = time.time() - start_time
    
    # Compute metrics
    if problem_type == 'classification':
        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        print(f"   ✅ Terminé en {train_time:.2f}s")
        print(f"   📈 Train Accuracy: {train_score:.4f}")
        print(f"   📉 Test Accuracy: {test_score:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🎯 Recall: {recall:.4f}")
        print(f"   🎯 F1-Score: {f1:.4f}")
        
        results.append({
            'Model': model_name,
            'Features': X_tr.shape[1],
            'Train_Accuracy': train_score,
            'Test_Accuracy': test_score,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Train_Time': train_time
        })
    else:
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"   ✅ Terminé en {train_time:.2f}s")
        print(f"   📈 Train R²: {train_score:.4f}")
        print(f"   📉 Test R²: {test_score:.4f}")
        print(f"   🎯 RMSE: {rmse:.4f}")
        print(f"   🎯 MAE: {mae:.4f}")
        
        results.append({
            'Model': model_name,
            'Features': X_tr.shape[1],
            'Train_R2': train_score,
            'Test_R2': test_score,
            'RMSE': rmse,
            'MAE': mae,
            'Train_Time': train_time
        })

print(f"\n{'='*60}")
print("✅ TOUS LES MODÈLES ENTRAÎNÉS!")
print(f"{'='*60}")

# ==============================================================================
# CELLULE 10: COMPARAISON ANALYTIQUE DES RÉSULTATS
# ==============================================================================

print("="*70)
print("📊 COMPARAISON ANALYTIQUE DES RÉSULTATS")
print("="*70)

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n📋 Tableau Complet des Résultats:")
display(results_df)

# Sort by test performance
if problem_type == 'classification':
    results_df_sorted = results_df.sort_values('Test_Accuracy', ascending=False)
    best_model = results_df_sorted.iloc[0]
    metric_name = 'Test Accuracy'
    metric_col = 'Test_Accuracy'
else:
    results_df_sorted = results_df.sort_values('Test_R2', ascending=False)
    best_model = results_df_sorted.iloc[0]
    metric_name = 'Test R²'
    metric_col = 'Test_R2'

print(f"\n🏆 MEILLEUR MODÈLE:")
print(f"   - Nom: {best_model['Model']}")
print(f"   - {metric_name}: {best_model[metric_col]:.4f}")
print(f"   - Features: {best_model['Features']}")
print(f"   - Temps: {best_model['Train_Time']:.2f}s")

# Find RLT position (in sorted dataframe) - FIX: use position in sorted, not original index!
rlt_score = None
rlt_position = None

for idx, model_name in enumerate(results_df_sorted['Model'].values):
    if 'RLT' in model_name:
        rlt_position = idx + 1  # Position in sorted ranking (1-indexed)
        rlt_score = results_df_sorted.iloc[idx][metric_col]
        break

# Fallback if not found
if rlt_position is None:
    rlt_position = len(results_df)
    rlt_score = 0

print(f"\n🌲 RLT-ExtraTrees:")
print(f"   - Position: #{rlt_position} / {len(results_df)}")
print(f"   - {metric_name}: {rlt_score:.4f}")
print(f"   - Features utilisées: {X_train_rlt.shape[1]} (original: {X_train.shape[1]})")

# Visualizations
print("\n📊 Génération des visualisations...")

# Plot 1: Performance comparison
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
colors = ['red' if 'RLT' in name else 'steelblue' for name in results_df_sorted['Model']]
plt.barh(range(len(results_df_sorted)), results_df_sorted[metric_col], color=colors)
plt.yticks(range(len(results_df_sorted)), results_df_sorted['Model'])
plt.xlabel(metric_name)
plt.title(f'Comparaison des Modèles - {metric_name}', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Plot 2: Training time
plt.subplot(1, 2, 2)
plt.barh(range(len(results_df_sorted)), results_df_sorted['Train_Time'], color=colors)
plt.yticks(range(len(results_df_sorted)), results_df_sorted['Model'])
plt.xlabel('Temps (secondes)')
plt.title('Temps d Entraînement', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\n📈 STATISTIQUES GLOBALES:")
print(f"   - Meilleur {metric_name}: {results_df[metric_col].max():.4f}")
print(f"   - Pire {metric_name}: {results_df[metric_col].min():.4f}")
print(f"   - Moyenne {metric_name}: {results_df[metric_col].mean():.4f}")
print(f"   - Écart-type: {results_df[metric_col].std():.4f}")

# Improvement analysis
rlt_score = results_df[results_df['Model'].str.contains('RLT')].iloc[0][metric_col]
best_other_score = results_df[~results_df['Model'].str.contains('RLT')][metric_col].max()
improvement = ((rlt_score - best_other_score) / best_other_score) * 100

print(f"\n🔍 ANALYSE RLT:")
if rlt_score > best_other_score:
    print(f"   ✅ RLT est MEILLEUR que les autres modèles")
    print(f"   📈 Amélioration: +{improvement:.2f}%")
elif abs(rlt_score - best_other_score) < 0.01:
    print(f"   ≈ RLT est ÉQUIVALENT aux autres modèles")
    print(f"   📊 Différence: {improvement:+.2f}%")
else:
    print(f"   ⚠️  RLT est moins performant")
    print(f"   📉 Différence: {improvement:.2f}%")

print(f"\n💡 CONCLUSION:")
print(f"   🏆 Meilleur modèle: {best_model['Model']}")
print(f"   📊 {metric_name}: {best_model[metric_col]:.4f}")
if rlt_position == 1:
    print(f"   🌲 RLT-ExtraTrees: GAGNANT! (#1/{len(results_df)})")
    print(f"   ✅ VI + Muting + Linear Combinations = Succès!")
elif rlt_position <= 3:
    print(f"   🌲 RLT-ExtraTrees: Très bon résultat (#{rlt_position}/{len(results_df)})")
    print(f"   📈 Performance compétitive avec {rlt_score:.4f}")
else:
    print(f"   🌲 RLT-ExtraTrees: #{rlt_position}/{len(results_df)}")
    print(f"   💡 Suggestion: Ajuster VI_THRESHOLD ou combinaisons")

print("\n" + "="*70)
print("✅ ANALYSE COMPLÈTE TERMINÉE!")
print("="*70)

# ==============================================================================
# CELLULE 11: SAUVEGARDE DES RÉSULTATS
# ==============================================================================

print("="*70)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*70)

# Save results to CSV
csv_filename = f"results_{filename.replace('.csv', '')}.csv"
results_df.to_csv(csv_filename, index=False)
print(f"\\n✅ Résultats sauvegardés: {csv_filename}")

# Download results
files.download(csv_filename)
print(f"📥 Téléchargement du fichier de résultats...")

print("\\n🎉 PROJET TERMINÉ!")
print(f"\\n📊 Résumé:")
print(f"   - Dataset: {filename}")
print(f"   - Type: {problem_type.upper()}")
print(f"   - Samples: {df.shape[0]}")
print(f"   - Features (origin): {df.shape[1] - 1}")
print(f"   - Features (after prep): {X_train.shape[1]}")
print(f"   - Features (RLT muted): {X_train_muted.shape[1]}")
print(f"   - Features (RLT + combinations): {X_train_rlt.shape[1]}")
print(f"   - Models entraînés: 8")
print(f"   - Meilleur modèle: {best_model['Model']}")
print(f"   - Meilleur {metric_name}: {best_model[metric_col]:.4f}")
print(f"\\n✅ Tous les résultats ont été sauvegardés!")
