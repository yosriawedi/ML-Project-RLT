"""
CRISP-DM STEP 3: DATA PREPARATION
Comprehensive data cleaning, encoding, scaling, and RLT-inspired feature selection
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, VarianceThreshold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Define workspace path
workspace_path = r'c:\Users\DELL\Downloads\(No subject)'

# Create output directory
prep_dir = os.path.join(workspace_path, 'prepared_data')
os.makedirs(prep_dir, exist_ok=True)

print("=" * 100)
print("CRISP-DM STEP 3: DATA PREPARATION")
print("=" * 100)
print("\n")

class RLTDataPreparator:
    """
    Data preparation with RLT-inspired variable importance and muting
    """
    
    def __init__(self, name, problem_type):
        self.name = name
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.original_features = None
        self.vi_scores = None
        self.muted_features = []
        self.kept_features = []
        
    def clean_column_names(self, df):
        """Clean column names"""
        print("\n📝 Cleaning column names...")
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        return df
    
    def handle_missing_values(self, df, target_col):
        """Handle missing values using multiple strategies"""
        print("\n🔧 Handling missing values...")
        
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found")
            return df
        
        print(f"Found {missing.sum()} missing values across {(missing > 0).sum()} columns")
        
        for col in df.columns:
            if col == target_col:
                continue
                
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Numerical: try mean, median, or mode based on distribution
                    skew = df[col].skew()
                    if abs(skew) > 1:
                        df[col].fillna(df[col].median(), inplace=True)
                        print(f"  • {col}: Filled with median (skewed)")
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)
                        print(f"  • {col}: Filled with mean")
                else:
                    # Categorical: fill with mode
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    print(f"  • {col}: Filled with mode")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\n🔧 Removing duplicates...")
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"✓ Removed {n_removed} duplicate rows ({n_removed/n_before*100:.2f}%)")
        else:
            print("✓ No duplicates found")
        return df
    
    def encode_categorical(self, df, target_col):
        """Encode categorical variables"""
        print("\n🔧 Encoding categorical variables...")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        if len(categorical_cols) == 0:
            print("✓ No categorical features to encode")
            return df
        
        print(f"Found {len(categorical_cols)} categorical features")
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique == 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  • {col}: Binary encoded")
            elif n_unique <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
                print(f"  • {col}: One-hot encoded ({n_unique} categories)")
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  • {col}: Label encoded ({n_unique} categories)")
        
        return df
    
    def encode_target(self, y):
        """Encode target variable for classification"""
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                self.label_encoders['target'] = le
                print(f"✓ Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                return y_encoded
        return y
    
    def remove_multicollinearity(self, X, threshold=0.95):
        """Remove highly correlated features"""
        print(f"\n🔧 Removing multicollinearity (threshold={threshold})...")
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        if len(to_drop) > 0:
            X = X.drop(columns=to_drop)
            print(f"✓ Removed {len(to_drop)} highly correlated features: {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
        else:
            print("✓ No severe multicollinearity detected")
        
        return X
    
    def compute_variable_importance(self, X, y):
        """
        Compute global variable importance using ensemble methods
        This is the core RLT concept: identify strong vs weak variables
        """
        print("\n💡 COMPUTING RLT-STYLE VARIABLE IMPORTANCE...")
        print("Using ensemble methods to estimate global VI")
        
        # Use multiple methods and aggregate
        vi_methods = {}
        
        # 1. Random Forest VI
        print("  • Random Forest VI...")
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        
        rf.fit(X, y)
        vi_methods['RandomForest'] = rf.feature_importances_
        
        # 2. Extra Trees VI (more randomization)
        print("  • Extra Trees VI...")
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            et = ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        else:
            et = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        
        et.fit(X, y)
        vi_methods['ExtraTrees'] = et.feature_importances_
        
        # 3. Statistical test VI
        print("  • Statistical test VI...")
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            selector = SelectKBest(f_classif, k='all')
        else:
            selector = SelectKBest(f_regression, k='all')
        
        selector.fit(X, y)
        scores_normalized = selector.scores_ / selector.scores_.sum()
        vi_methods['Statistical'] = scores_normalized
        
        # Aggregate VI scores (weighted average)
        weights = {'RandomForest': 0.4, 'ExtraTrees': 0.4, 'Statistical': 0.2}
        vi_aggregate = np.zeros(X.shape[1])
        
        for method, weight in weights.items():
            vi_aggregate += weights[method] * vi_methods[method]
        
        # Create VI DataFrame
        self.vi_scores = pd.DataFrame({
            'Feature': X.columns,
            'VI_Aggregate': vi_aggregate,
            'VI_RF': vi_methods['RandomForest'],
            'VI_ET': vi_methods['ExtraTrees'],
            'VI_Stat': vi_methods['Statistical']
        }).sort_values('VI_Aggregate', ascending=False)
        
        print("\n" + "=" * 80)
        print("VARIABLE IMPORTANCE RANKING (Top 20):")
        print("=" * 80)
        print(self.vi_scores.head(20).to_string(index=False))
        
        return self.vi_scores
    
    def apply_variable_muting(self, X, vi_threshold=0.01, top_k=None, percentile=None):
        """
        Apply RLT variable muting: progressively eliminate noise variables
        
        Args:
            vi_threshold: Minimum VI score to keep feature
            top_k: Keep only top k features
            percentile: Keep features above this VI percentile
        """
        print("\n🔇 APPLYING RLT VARIABLE MUTING...")
        
        muted_features = []
        kept_features = []
        
        # Strategy 1: VI threshold
        if vi_threshold is not None:
            low_vi = self.vi_scores[self.vi_scores['VI_Aggregate'] < vi_threshold]
            muted_features.extend(low_vi['Feature'].tolist())
            print(f"  • Threshold strategy (VI < {vi_threshold}): {len(low_vi)} features muted")
        
        # Strategy 2: Top-k selection
        if top_k is not None and top_k < len(self.vi_scores):
            kept_features = self.vi_scores.head(top_k)['Feature'].tolist()
            muted_features = [f for f in X.columns if f not in kept_features]
            print(f"  • Top-k strategy (k={top_k}): {len(muted_features)} features muted")
        
        # Strategy 3: Percentile cutoff
        if percentile is not None:
            vi_cutoff = np.percentile(self.vi_scores['VI_Aggregate'], percentile)
            low_vi = self.vi_scores[self.vi_scores['VI_Aggregate'] < vi_cutoff]
            muted_features.extend(low_vi['Feature'].tolist())
            print(f"  • Percentile strategy ({percentile}th percentile): {len(low_vi)} features muted")
        
        # Remove duplicates
        muted_features = list(set(muted_features))
        kept_features = [f for f in X.columns if f not in muted_features]
        
        self.muted_features = muted_features
        self.kept_features = kept_features
        
        print(f"\n✓ MUTING SUMMARY:")
        print(f"  • Original features: {len(X.columns)}")
        print(f"  • Muted features: {len(muted_features)} ({len(muted_features)/len(X.columns)*100:.1f}%)")
        print(f"  • Kept features: {len(kept_features)} ({len(kept_features)/len(X.columns)*100:.1f}%)")
        
        if len(muted_features) > 0:
            print(f"\n  Muted features: {muted_features[:10]}{'...' if len(muted_features) > 10 else ''}")
        
        # Return X with only kept features
        X_muted = X[kept_features]
        
        return X_muted
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        print("\n📏 Scaling features (StandardScaler)...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print(f"✓ Scaled {X.shape[1]} features")
        
        return X_scaled
    
    def prepare(self, df, target_col, apply_muting=True, vi_threshold=0.01):
        """Complete preparation pipeline"""
        
        print("\n" + "=" * 100)
        print(f"PREPARING: {self.name}")
        print("=" * 100)
        
        # 1. Clean column names
        df = self.clean_column_names(df)
        import re
        target_col = target_col.strip().replace(' ', '_')
        target_col = re.sub(r'[^a-zA-Z0-9_]', '', target_col)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df, target_col)
        
        # 3. Remove duplicates
        df = self.remove_duplicates(df)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.original_features = X.columns.tolist()
        
        # 4. Encode categorical features
        X = self.encode_categorical(X, target_col)
        
        # 5. Encode target
        y = self.encode_target(y)
        
        # 6. Remove multicollinearity
        X = self.remove_multicollinearity(X, threshold=0.95)
        
        # 7. Compute variable importance (RLT Step 1)
        vi_scores = self.compute_variable_importance(X, y)
        
        # 8. Save full prepared dataset (before muting)
        X_full = X.copy()
        
        # 9. Apply variable muting (RLT Step 2) - optional
        if apply_muting:
            X = self.apply_variable_muting(X, vi_threshold=vi_threshold)
        
        # 10. Scale features
        X = self.scale_features(X, fit=True)
        
        print("\n" + "=" * 80)
        print("PREPARATION SUMMARY:")
        print("=" * 80)
        print(f"✓ Original shape: {df.shape}")
        print(f"✓ Prepared shape: {X.shape}")
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Target distribution: {np.unique(y, return_counts=True)}")
        
        return X, y, X_full, vi_scores


def to_numpy(data):
    """Convert data to numpy array safely"""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'values'):
        return data.values
    else:
        return np.array(data)


def prepare_all_datasets():
    """Prepare all datasets"""
    
    results = []
    
    # 1. Boston Housing
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'BostonHousing.csv'))
        prep = RLTDataPreparator('BostonHousing', 'REGRESSION')
        X, y, X_full, vi = prep.prepare(df, 'medv', apply_muting=True, vi_threshold=0.01)
        
        # Save
        np.save(os.path.join(prep_dir, 'BostonHousing_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'BostonHousing_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'BostonHousing_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'BostonHousing_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'BostonHousing_prep.pkl'), 'wb'))
        
        results.append({'name': 'BostonHousing', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'BostonHousing', 'status': 'FAILED', 'error': str(e)})
    
    # 2. Wine Quality Red
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'winequality-red.csv'), sep=';')
        prep = RLTDataPreparator('WineQuality_Red', 'CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'quality', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'WineQuality_Red_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'WineQuality_Red_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'WineQuality_Red_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'WineQuality_Red_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'WineQuality_Red_prep.pkl'), 'wb'))
        
        results.append({'name': 'WineQuality_Red', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'WineQuality_Red', 'status': 'FAILED', 'error': str(e)})
    
    # 3. Wine Quality White
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'winequality-white.csv'), sep=';')
        prep = RLTDataPreparator('WineQuality_White', 'CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'quality', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'WineQuality_White_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'WineQuality_White_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'WineQuality_White_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'WineQuality_White_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'WineQuality_White_prep.pkl'), 'wb'))
        
        results.append({'name': 'WineQuality_White', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'WineQuality_White', 'status': 'FAILED', 'error': str(e)})
    
    # 4. Sonar
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'sonar data.csv'), header=None)
        df.columns = [f'feature_{i}' for i in range(60)] + ['target']
        prep = RLTDataPreparator('Sonar', 'BINARY CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'target', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'Sonar_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'Sonar_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'Sonar_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'Sonar_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'Sonar_prep.pkl'), 'wb'))
        
        results.append({'name': 'Sonar', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'Sonar', 'status': 'FAILED', 'error': str(e)})
    
    # 5. Parkinsons
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'parkinsons.data'))
        df = df.drop('name', axis=1)
        prep = RLTDataPreparator('Parkinsons', 'BINARY CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'status', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'Parkinsons_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'Parkinsons_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'Parkinsons_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'Parkinsons_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'Parkinsons_prep.pkl'), 'wb'))
        
        results.append({'name': 'Parkinsons', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'Parkinsons', 'status': 'FAILED', 'error': str(e)})
    
    # 6. WDBC Breast Cancer
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'wdbc.data'), header=None)
        feature_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)]
        df.columns = feature_names
        df = df.drop('id', axis=1)
        prep = RLTDataPreparator('WDBC_BreastCancer', 'BINARY CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'diagnosis', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'WDBC_BreastCancer_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'WDBC_BreastCancer_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'WDBC_BreastCancer_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'WDBC_BreastCancer_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'WDBC_BreastCancer_prep.pkl'), 'wb'))
        
        results.append({'name': 'WDBC_BreastCancer', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'WDBC_BreastCancer', 'status': 'FAILED', 'error': str(e)})
    
    # 7. Auto MPG
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'auto-mpg.data'), 
                         delim_whitespace=True, header=None,
                         names=['mpg', 'cylinders', 'displacement', 'horsepower', 
                                'weight', 'acceleration', 'model_year', 'origin', 'car_name'])
        df = df.drop('car_name', axis=1)
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        prep = RLTDataPreparator('AutoMPG', 'REGRESSION')
        X, y, X_full, vi = prep.prepare(df, 'mpg', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'AutoMPG_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'AutoMPG_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'AutoMPG_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'AutoMPG_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'AutoMPG_prep.pkl'), 'wb'))
        
        results.append({'name': 'AutoMPG', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'AutoMPG', 'status': 'FAILED', 'error': str(e)})
    
    # 8. School Data
    print("\n" + "▶" * 50)
    try:
        df = pd.read_csv(os.path.join(workspace_path, 'data_school.csv'))
        prep = RLTDataPreparator('SchoolData', 'CLASSIFICATION')
        X, y, X_full, vi = prep.prepare(df, 'Target', apply_muting=True, vi_threshold=0.01)
        
        np.save(os.path.join(prep_dir, 'SchoolData_X.npy'), to_numpy(X))
        np.save(os.path.join(prep_dir, 'SchoolData_y.npy'), to_numpy(y))
        np.save(os.path.join(prep_dir, 'SchoolData_X_full.npy'), to_numpy(X_full))
        vi.to_csv(os.path.join(prep_dir, 'SchoolData_VI.csv'), index=False)
        pickle.dump(prep, open(os.path.join(prep_dir, 'SchoolData_prep.pkl'), 'wb'))
        
        results.append({'name': 'SchoolData', 'status': 'SUCCESS', 'shape': X.shape})
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'name': 'SchoolData', 'status': 'FAILED', 'error': str(e)})
    
    return results


# Run preparation
results = prepare_all_datasets()

# Final Summary
print("\n" + "=" * 100)
print("STEP 3 COMPLETE: DATA PREPARATION SUMMARY")
print("=" * 100)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

print(f"\n✓ Prepared datasets saved to: {prep_dir}")
print(f"✓ Files saved:")
print("  • X.npy: Muted features (RLT-style)")
print("  • X_full.npy: All features (baseline comparison)")
print("  • y.npy: Target variable")
print("  • VI.csv: Variable importance scores")
print("  • prep.pkl: Preprocessor object")

print("\n" + "=" * 100)
print("STEP 3 COMPLETE ✓")
print("=" * 100)
print("\nNext: STEP 4 - MODELING")
