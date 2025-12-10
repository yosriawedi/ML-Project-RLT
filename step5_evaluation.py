"""
CRISP-DM STEP 5: EVALUATION
Comprehensive model evaluation with confusion matrices, ROC curves, feature importance, and error analysis
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                             roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
                             mean_squared_error, mean_absolute_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Workspace path
workspace_path = r'c:\Users\DELL\Downloads\(No subject)'
prep_dir = os.path.join(workspace_path, 'prepared_data')
models_dir = os.path.join(workspace_path, 'models')
eval_dir = os.path.join(workspace_path, 'evaluation')
os.makedirs(eval_dir, exist_ok=True)

print("=" * 100)
print("CRISP-DM STEP 5: EVALUATION")
print("=" * 100)
print("\n")


def plot_confusion_matrix(y_true, y_pred, dataset_name, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=True)
    plt.title(f'{dataset_name}: {model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'{dataset_name}_{model_name}_confusion_matrix.png'), dpi=100)
    plt.close()


def plot_roc_curve(y_true, y_proba, dataset_name, model_name):
    """Plot ROC curve for binary classification"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{dataset_name}: {model_name}\nROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'{dataset_name}_{model_name}_roc_curve.png'), dpi=100)
    plt.close()


def plot_regression_residuals(y_true, y_pred, dataset_name, model_name):
    """Plot residuals for regression"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Actual vs Predicted
    axes[1].scatter(y_true, y_pred, alpha=0.5, color='green')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'{dataset_name}: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'{dataset_name}_{model_name}_residuals.png'), dpi=100)
    plt.close()


def evaluate_classification(dataset_name, problem_type, X, y, model, model_name):
    """Comprehensive evaluation for classification"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {dataset_name} - {model_name}")
    print(f"{'='*80}")
    
    # Train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    
    # ROC-AUC for binary classification
    if len(np.unique(y)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"  • ROC-AUC:   {roc_auc:.4f}")
            
            # Plot ROC curve
            plot_roc_curve(y_test, y_proba, dataset_name, model_name)
            print(f"  ✓ ROC curve saved")
        except:
            pass
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, dataset_name, model_name)
    print(f"  ✓ Confusion matrix saved")
    
    # Classification Report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        n_features = len(importance)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Top 10 features
        indices = np.argsort(importance)[::-1][:10]
        print(f"\n🔝 TOP 10 IMPORTANT FEATURES:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Error Analysis
    errors = y_test != y_pred
    error_rate = errors.sum() / len(y_test)
    print(f"\n❌ ERROR ANALYSIS:")
    print(f"  • Total Errors: {errors.sum()}/{len(y_test)} ({error_rate*100:.2f}%)")
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'error_rate': error_rate
    }


def evaluate_regression(dataset_name, X, y, model, model_name):
    """Comprehensive evaluation for regression"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {dataset_name} - {model_name}")
    print(f"{'='*80}")
    
    # Train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📊 REGRESSION METRICS:")
    print(f"  • R² Score:  {r2:.4f}")
    print(f"  • RMSE:      {rmse:.4f}")
    print(f"  • MAE:       {mae:.4f}")
    print(f"  • MAPE:      {mape:.2f}%")
    
    # Residual plots
    plot_regression_residuals(y_test, y_pred, dataset_name, model_name)
    print(f"  ✓ Residual plots saved")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        n_features = len(importance)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Top 10 features
        indices = np.argsort(importance)[::-1][:10]
        print(f"\n🔝 TOP 10 IMPORTANT FEATURES:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Error Analysis
    errors = np.abs(y_test - y_pred)
    large_errors = errors > (2 * rmse)
    print(f"\n❌ ERROR ANALYSIS:")
    print(f"  • Mean Absolute Error: {mae:.4f}")
    print(f"  • Large Errors (>2×RMSE): {large_errors.sum()}/{len(y_test)} ({large_errors.sum()/len(y_test)*100:.2f}%)")
    print(f"  • Max Error: {errors.max():.4f}")
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def evaluate_all_datasets():
    """Evaluate best models for all datasets"""
    
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
    
    evaluation_results = []
    
    for dataset_name, problem_type in datasets:
        print("\n" + "▶" * 50)
        
        try:
            # Load data
            X = np.load(os.path.join(prep_dir, f'{dataset_name}_X.npy'))
            y = np.load(os.path.join(prep_dir, f'{dataset_name}_y.npy'))
            
            # Load best model
            model = pickle.load(open(os.path.join(models_dir, f'{dataset_name}_best_model.pkl'), 'rb'))
            
            # Load results to get model name
            results_df = pd.read_csv(os.path.join(models_dir, f'{dataset_name}_results.csv'))
            best_idx = results_df['primary_metric'].idxmax()
            model_name = results_df.loc[best_idx, 'model']
            
            print(f"\n🏆 Best Model: {model_name}")
            
            # Evaluate
            if problem_type in ['BINARY CLASSIFICATION', 'CLASSIFICATION']:
                result = evaluate_classification(dataset_name, problem_type, X, y, model, model_name)
            else:
                result = evaluate_regression(dataset_name, X, y, model, model_name)
            
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save evaluation results
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv(os.path.join(eval_dir, 'evaluation_results.csv'), index=False)
        
        print("\n" + "=" * 100)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 100)
        print(eval_df.to_string(index=False))
    
    return evaluation_results


# Run evaluation
print("Starting comprehensive evaluation...")
print("\n")

results = evaluate_all_datasets()

print("\n" + "=" * 100)
print("STEP 5 COMPLETE ✓")
print("=" * 100)
print(f"\n✓ Evaluation results saved to: {eval_dir}")
print("✓ Generated:")
print("  • Confusion matrices for classification")
print("  • ROC curves for binary classification")
print("  • Residual plots for regression")
print("  • Feature importance analysis")
print("  • Error analysis reports")
print("\nNext: STEP 6 - DEPLOYMENT")
