"""
PRODUCTION-READY ML PIPELINE WITH RLT METHODOLOGY
Complete pipeline for data preprocessing, training, prediction, and model persistence
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, classification_report


class RLTMLPipeline:
    """
    Production ML Pipeline with Reinforcement Learning Trees (RLT) methodology
    
    Features:
    - Automated data preprocessing
    - RLT-style variable importance and muting
    - Model training with cross-validation
    - Prediction on new data
    - Model persistence (save/load)
    """
    
    def __init__(self, problem_type='classification', vi_threshold=0.01):
        """
        Initialize pipeline
        
        Args:
            problem_type: 'classification' or 'regression'
            vi_threshold: Variable importance threshold for muting (default 0.01)
        """
        self.problem_type = problem_type.lower()
        self.vi_threshold = vi_threshold
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
        self.model = None
        self.vi_scores = None
        self.kept_features = []
        self.feature_names = []
        
    def preprocess(self, df, target_col, fit=True):
        """
        Preprocess data: handle missing values, encode categoricals, scale features
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            fit: If True, fit encoders and scalers; if False, use existing
        
        Returns:
            X, y: Processed features and target
        """
        print(f"\n{'='*60}")
        print("PREPROCESSING DATA")
        print(f"{'='*60}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Handle missing values
        print("• Handling missing values...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        # Encode categorical features
        print("• Encoding categorical features...")
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Encode target (for classification)
        if self.problem_type == 'classification':
            if y.dtype == 'object':
                if fit:
                    self.target_encoder = LabelEncoder()
                    y = self.target_encoder.fit_transform(y)
                else:
                    if self.target_encoder:
                        y = self.target_encoder.transform(y)
        
        # Scale features
        print("• Scaling features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"✓ Preprocessing complete: {X_scaled.shape}")
        
        return X_scaled, y
    
    def compute_variable_importance(self, X, y):
        """
        Compute RLT-style variable importance using ensemble methods
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            DataFrame with VI scores
        """
        print(f"\n{'='*60}")
        print("COMPUTING VARIABLE IMPORTANCE (RLT)")
        print(f"{'='*60}")
        
        # Use Random Forest for VI estimation
        if self.problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        
        estimator.fit(X, y)
        importance = estimator.feature_importances_
        
        # Create VI DataFrame
        self.vi_scores = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Features:")
        print(self.vi_scores.head(10).to_string(index=False))
        
        return self.vi_scores
    
    def apply_variable_muting(self, X):
        """
        Apply RLT variable muting: remove low-importance features
        
        Args:
            X: Feature matrix
        
        Returns:
            X_muted: Feature matrix with only high-importance features
        """
        print(f"\n{'='*60}")
        print("APPLYING RLT VARIABLE MUTING")
        print(f"{'='*60}")
        
        # Identify features to keep
        high_vi_features = self.vi_scores[self.vi_scores['Importance'] >= self.vi_threshold]
        self.kept_features = high_vi_features['Feature'].tolist()
        
        muted_count = len(X.columns) - len(self.kept_features)
        muted_pct = (muted_count / len(X.columns)) * 100
        
        print(f"• Original features: {len(X.columns)}")
        print(f"• Muted features: {muted_count} ({muted_pct:.1f}%)")
        print(f"• Kept features: {len(self.kept_features)} ({100-muted_pct:.1f}%)")
        
        # Return muted feature set
        X_muted = X[self.kept_features]
        
        print(f"✓ Variable muting complete")
        
        return X_muted
    
    def train(self, X, y, apply_muting=True):
        """
        Train RLT-style model with optional variable muting
        
        Args:
            X: Feature matrix
            y: Target vector
            apply_muting: If True, apply variable muting
        
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}")
        
        # Compute variable importance
        self.compute_variable_importance(X, y)
        
        # Apply variable muting (optional)
        if apply_muting:
            X = self.apply_variable_muting(X)
        else:
            self.kept_features = X.columns.tolist()
        
        # Select model
        if self.problem_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        
        # Train model
        print(f"\n• Training {self.model.__class__.__name__}...")
        self.model.fit(X, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✓ Training complete")
        print(f"• CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
        
        Returns:
            predictions: Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Keep only selected features
        X_selected = X[self.kept_features]
        
        # Make predictions
        predictions = self.model.predict(X_selected)
        
        # Decode predictions if classification
        if self.problem_type == 'classification' and self.target_encoder:
            predictions = self.target_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (classification only)
        
        Args:
            X: Feature matrix
        
        Returns:
            probabilities: Prediction probabilities
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Keep only selected features
        X_selected = X[self.kept_features]
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_selected)
        
        return probabilities
    
    def save_model(self, filepath):
        """
        Save trained pipeline to disk
        
        Args:
            filepath: Path to save pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load trained pipeline from disk
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            Loaded pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✓ Model loaded from: {filepath}")
        return pipeline


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*80)
    print("RLT ML PIPELINE - EXAMPLE USAGE")
    print("="*80)
    
    # Example: Classification task
    print("\n📋 EXAMPLE: Classification Task")
    print("-"*60)
    
    # Create sample data (replace with your actual data)
    from sklearn.datasets import make_classification
    X_sample, y_sample = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, n_classes=3, random_state=42
    )
    
    df_sample = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(20)])
    df_sample['target'] = y_sample
    
    # Initialize pipeline
    pipeline = RLTMLPipeline(problem_type='classification', vi_threshold=0.01)
    
    # Preprocess data
    X_processed, y_processed = pipeline.preprocess(df_sample, target_col='target', fit=True)
    
    # Train model with RLT variable muting
    model = pipeline.train(X_processed, y_processed, apply_muting=True)
    
    # Make predictions
    predictions = pipeline.predict(X_processed.head(10))
    print(f"\n📊 Sample Predictions: {predictions[:5]}")
    
    # Get probabilities
    probabilities = pipeline.predict_proba(X_processed.head(10))
    print(f"📊 Sample Probabilities:\n{probabilities[:3]}")
    
    # Save model
    pipeline.save_model('rlt_model.pkl')
    
    # Load model
    loaded_pipeline = RLTMLPipeline.load_model('rlt_model.pkl')
    
    print("\n" + "="*80)
    print("✓ PIPELINE EXAMPLE COMPLETE")
    print("="*80)
    print("\nThe pipeline is production-ready and can be deployed!")
    print("Key features:")
    print("  • Automated preprocessing")
    print("  • RLT variable importance & muting")
    print("  • Model training with CV")
    print("  • Prediction on new data")
    print("  • Model persistence (save/load)")
