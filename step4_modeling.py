"""
CRISP-DM STEP 4: MODELING
Baseline models vs RLT-style models with variable muting
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
import pickle
import time

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - will be skipped")

# Workspace path
workspace_path = r'c:\Users\DELL\Downloads\(No subject)'
prep_dir = os.path.join(workspace_path, 'prepared_data')
models_dir = os.path.join(workspace_path, 'models')
os.makedirs(models_dir, exist_ok=True)

print("=" * 100)
print("CRISP-DM STEP 4: MODELING")
print("=" * 100)
print("\n")


class RLTModelingPipeline:
    """
    Complete modeling pipeline comparing baseline models with RLT-style models
    """
    
    def __init__(self, name, problem_type):
        self.name = name
        self.problem_type = problem_type
        self.results = []
        self.best_model = None
        self.best_score = -np.inf
        
    def get_baseline_models(self):
        """Get baseline models for comparison"""
        
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss')
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=0.1, random_state=42, max_iter=2000),
                'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, max_depth=5)
        
        return models
    
    def get_rlt_models(self):
        """
        Get RLT-style models
        These use the muted feature set (high VI features only)
        """
        
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            models = {
                'RLT-RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'RLT-ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'RLT-GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
            }
            if XGBOOST_AVAILABLE:
                models['RLT-XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss')
        else:
            models = {
                'RLT-RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'RLT-ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
                'RLT-GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
            if XGBOOST_AVAILABLE:
                models['RLT-XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, max_depth=5)
        
        return models
    
    def evaluate_model(self, model, X, y, cv=5):
        """Evaluate model using cross-validation"""
        
        if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
            # Classification metrics
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Accuracy
            accuracy = cross_val_score(model, X, y, cv=cv_splitter, scoring='accuracy', n_jobs=-1)
            
            # F1 score
            f1 = cross_val_score(model, X, y, cv=cv_splitter, scoring='f1_weighted', n_jobs=-1)
            
            # Try ROC-AUC for binary classification
            try:
                if len(np.unique(y)) == 2:
                    roc_auc = cross_val_score(model, X, y, cv=cv_splitter, scoring='roc_auc', n_jobs=-1)
                else:
                    roc_auc = cross_val_score(model, X, y, cv=cv_splitter, scoring='roc_auc_ovr_weighted', n_jobs=-1)
            except:
                roc_auc = np.array([0])
            
            return {
                'accuracy_mean': accuracy.mean(),
                'accuracy_std': accuracy.std(),
                'f1_mean': f1.mean(),
                'f1_std': f1.std(),
                'roc_auc_mean': roc_auc.mean(),
                'roc_auc_std': roc_auc.std(),
                'primary_metric': accuracy.mean()
            }
        else:
            # Regression metrics
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # R2 score
            r2 = cross_val_score(model, X, y, cv=cv_splitter, scoring='r2', n_jobs=-1)
            
            # Negative MSE (sklearn returns negative)
            neg_mse = cross_val_score(model, X, y, cv=cv_splitter, scoring='neg_mean_squared_error', n_jobs=-1)
            rmse = np.sqrt(-neg_mse)
            
            # Negative MAE
            neg_mae = cross_val_score(model, X, y, cv=cv_splitter, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -neg_mae
            
            return {
                'r2_mean': r2.mean(),
                'r2_std': r2.std(),
                'rmse_mean': rmse.mean(),
                'rmse_std': rmse.std(),
                'mae_mean': mae.mean(),
                'mae_std': mae.std(),
                'primary_metric': r2.mean()
            }
    
    def train_and_evaluate(self, X_full, X_muted, y):
        """Train and evaluate all models"""
        
        print("\n" + "=" * 100)
        print(f"MODELING: {self.name}")
        print("=" * 100)
        print(f"Problem Type: {self.problem_type}")
        print(f"Full Features: {X_full.shape[1]}")
        print(f"Muted Features: {X_muted.shape[1]}")
        print(f"Samples: {X_full.shape[0]}")
        print(f"Feature Reduction: {(1 - X_muted.shape[1]/X_full.shape[1])*100:.1f}%")
        
        # PHASE 1: Baseline Models (using full feature set)
        print("\n" + "-" * 80)
        print("PHASE 1: BASELINE MODELS (Full Feature Set)")
        print("-" * 80)
        
        baseline_models = self.get_baseline_models()
        
        for model_name, model in baseline_models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            try:
                metrics = self.evaluate_model(model, X_full, y, cv=5)
                train_time = time.time() - start_time
                
                # Fit on full data for later use
                model.fit(X_full, y)
                
                result = {
                    'dataset': self.name,
                    'model': model_name,
                    'model_type': 'BASELINE',
                    'n_features': X_full.shape[1],
                    'train_time': train_time,
                    **metrics
                }
                
                self.results.append(result)
                
                # Update best model
                if metrics['primary_metric'] > self.best_score:
                    self.best_score = metrics['primary_metric']
                    self.best_model = (model_name, model, 'BASELINE')
                
                # Print results
                if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
                    print(f"  ✓ Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
                    print(f"  ✓ F1 Score: {metrics['f1_mean']:.4f} (±{metrics['f1_std']:.4f})")
                    if metrics['roc_auc_mean'] > 0:
                        print(f"  ✓ ROC-AUC: {metrics['roc_auc_mean']:.4f} (±{metrics['roc_auc_std']:.4f})")
                else:
                    print(f"  ✓ R²: {metrics['r2_mean']:.4f} (±{metrics['r2_std']:.4f})")
                    print(f"  ✓ RMSE: {metrics['rmse_mean']:.4f} (±{metrics['rmse_std']:.4f})")
                    print(f"  ✓ MAE: {metrics['mae_mean']:.4f} (±{metrics['mae_std']:.4f})")
                print(f"  ⏱️ Time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # PHASE 2: RLT-Style Models (using muted feature set)
        print("\n" + "-" * 80)
        print("PHASE 2: RLT-STYLE MODELS (Muted Feature Set - High VI Only)")
        print("-" * 80)
        print(f"RLT Variable Muting: Kept {X_muted.shape[1]}/{X_full.shape[1]} features ({X_muted.shape[1]/X_full.shape[1]*100:.1f}%)")
        
        rlt_models = self.get_rlt_models()
        
        for model_name, model in rlt_models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            try:
                metrics = self.evaluate_model(model, X_muted, y, cv=5)
                train_time = time.time() - start_time
                
                # Fit on full data for later use
                model.fit(X_muted, y)
                
                result = {
                    'dataset': self.name,
                    'model': model_name,
                    'model_type': 'RLT',
                    'n_features': X_muted.shape[1],
                    'train_time': train_time,
                    **metrics
                }
                
                self.results.append(result)
                
                # Update best model
                if metrics['primary_metric'] > self.best_score:
                    self.best_score = metrics['primary_metric']
                    self.best_model = (model_name, model, 'RLT')
                
                # Print results
                if self.problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
                    print(f"  ✓ Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
                    print(f"  ✓ F1 Score: {metrics['f1_mean']:.4f} (±{metrics['f1_std']:.4f})")
                    if metrics['roc_auc_mean'] > 0:
                        print(f"  ✓ ROC-AUC: {metrics['roc_auc_mean']:.4f} (±{metrics['roc_auc_std']:.4f})")
                else:
                    print(f"  ✓ R²: {metrics['r2_mean']:.4f} (±{metrics['r2_std']:.4f})")
                    print(f"  ✓ RMSE: {metrics['rmse_mean']:.4f} (±{metrics['rmse_std']:.4f})")
                    print(f"  ✓ MAE: {metrics['mae_mean']:.4f} (±{metrics['mae_std']:.4f})")
                print(f"  ⏱️ Time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Summary
        print("\n" + "=" * 80)
        print("MODELING SUMMARY")
        print("=" * 80)
        results_df = pd.DataFrame(self.results)
        print(results_df[['model', 'model_type', 'n_features', 'primary_metric', 'train_time']].to_string(index=False))
        
        print(f"\n🏆 BEST MODEL: {self.best_model[0]} ({self.best_model[2]})")
        print(f"   Score: {self.best_score:.4f}")
        
        return results_df


def model_all_datasets():
    """Model all prepared datasets"""
    
    all_results = []
    
    # Dataset configurations
    datasets = [
        ('BostonHousing', 'REGRESSION'),
        ('WineQuality_Red', 'CLASSIFICATION'),
        ('WineQuality_White', 'CLASSIFICATION'),
        ('Sonar', 'BINARY CLASSIFICATION'),
        ('Parkinsons', 'BINARY CLASSIFICATION'),
        ('WDBC_BreastCancer', 'BINARY CLASSIFICATION'),
        ('AutoMPG', 'REGRESSION'),
        ('SchoolData', 'CLASSIFICATION')
    ]
    
    for dataset_name, problem_type in datasets:
        print("\n" + "▶" * 50)
        
        try:
            # Load prepared data
            X_full = np.load(os.path.join(prep_dir, f'{dataset_name}_X_full.npy'))
            X_muted = np.load(os.path.join(prep_dir, f'{dataset_name}_X.npy'))
            y = np.load(os.path.join(prep_dir, f'{dataset_name}_y.npy'))
            
            # Create pipeline and run modeling
            pipeline = RLTModelingPipeline(dataset_name, problem_type)
            results_df = pipeline.train_and_evaluate(X_full, X_muted, y)
            
            # Save results
            results_df.to_csv(os.path.join(models_dir, f'{dataset_name}_results.csv'), index=False)
            
            # Save best model
            pickle.dump(pipeline.best_model[1], 
                       open(os.path.join(models_dir, f'{dataset_name}_best_model.pkl'), 'wb'))
            
            # Aggregate results
            all_results.append(results_df)
            
            print(f"\n✓ {dataset_name} modeling complete")
            print(f"✓ Results saved to: {dataset_name}_results.csv")
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(os.path.join(models_dir, 'ALL_RESULTS.csv'), index=False)
        
        print("\n" + "=" * 100)
        print("STEP 4 COMPLETE: ALL MODELING RESULTS")
        print("=" * 100)
        
        # Analysis: Baseline vs RLT comparison
        print("\n" + "=" * 80)
        print("BASELINE vs RLT COMPARISON")
        print("=" * 80)
        
        for dataset_name, _ in datasets:
            dataset_results = combined_results[combined_results['dataset'] == dataset_name]
            if len(dataset_results) > 0:
                baseline_best = dataset_results[dataset_results['model_type'] == 'BASELINE']['primary_metric'].max()
                rlt_best = dataset_results[dataset_results['model_type'] == 'RLT']['primary_metric'].max()
                
                improvement = ((rlt_best - baseline_best) / baseline_best) * 100
                winner = "RLT" if rlt_best > baseline_best else "BASELINE"
                
                print(f"\n{dataset_name}:")
                print(f"  Baseline Best: {baseline_best:.4f}")
                print(f"  RLT Best:      {rlt_best:.4f}")
                print(f"  Improvement:   {improvement:+.2f}%")
                print(f"  Winner:        {winner} {'🏆' if winner == 'RLT' else ''}")
        
        return combined_results
    
    return None


# Run modeling
print("Starting comprehensive modeling pipeline...")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")
print("\n")

results = model_all_datasets()

print("\n" + "=" * 100)
print("STEP 4 COMPLETE ✓")
print("=" * 100)
print(f"\n✓ All model results saved to: {models_dir}")
print("\nNext: STEP 5 - EVALUATION")
