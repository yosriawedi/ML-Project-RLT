"""
PRODUCTION-READY ML PIPELINE WITH RLT METHODOLOGY
Complete pipeline for data preprocessing, training, prediction, and persistence
"""

import warnings
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


class RLTMLPipeline:
    """ML Pipeline with Reinforcement Learning Trees (RLT) methodology."""

    def __init__(self, problem_type="classification", vi_threshold=0.01):
        self.problem_type = problem_type.lower()
        self.vi_threshold = vi_threshold
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
        self.model = None
        self.vi_scores = None
        self.kept_features = []
        self.feature_names = []

    def preprocess(self, df, target_col=None, fit=True):
        """Preprocess data: handle missing values, encode categoricals, scale features."""
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)

        # Séparation X / y seulement si target_col est fourni
        if target_col is not None:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None

        if fit:
            self.feature_names = X.columns.tolist()

        print("• Handling missing values...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ["int64", "float64"]:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)

        print("• Encoding categorical features...")
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col])

        # Encodage de la cible uniquement si target_col est fourni
        if (
            target_col is not None
            and self.problem_type == "classification"
            and y.dtype == "object"
        ):
            if fit:
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
            elif self.target_encoder:
                y = self.target_encoder.transform(y)

        print("• Scaling features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        print(f"✓ Preprocessing complete: {X_scaled.shape}")
        return X_scaled, y

    def compute_variable_importance(self, X, y):
        """Compute variable importance using Random Forest."""
        print("\n" + "=" * 60)
        print("COMPUTING VARIABLE IMPORTANCE (RLT)")
        print("=" * 60)

        estimator = (
            RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            )
            if self.problem_type == "classification"
            else RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            )
        )

        estimator.fit(X, y)
        importance = estimator.feature_importances_

        self.vi_scores = pd.DataFrame(
            {"Feature": X.columns, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        print("\nTop 10 Features:")
        print(self.vi_scores.head(10).to_string(index=False))
        return self.vi_scores

    def apply_variable_muting(self, X):
        """Keep only features above importance threshold."""
        print("\n" + "=" * 60)
        print("APPLYING RLT VARIABLE MUTING")
        print("=" * 60)

        high_vi = self.vi_scores[self.vi_scores["Importance"] >= self.vi_threshold]
        self.kept_features = high_vi["Feature"].tolist()

        muted_count = len(X.columns) - len(self.kept_features)
        muted_pct = (muted_count / len(X.columns)) * 100

        print(f"• Original features: {len(X.columns)}")
        print(f"• Muted features: {muted_count} ({muted_pct:.1f}%)")
        print(f"• Kept features: {len(self.kept_features)} ({100 - muted_pct:.1f}%)")

        X_muted = X[self.kept_features]
        print("✓ Variable muting complete")
        return X_muted

    def train(self, X, y, apply_muting=True):
        """Train RLT model with optional variable muting."""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

        self.compute_variable_importance(X, y)
        if apply_muting:
            X = self.apply_variable_muting(X)
        else:
            self.kept_features = X.columns.tolist()

        self.model = (
            RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            )
            if self.problem_type == "classification"
            else RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            )
        )

        print(f"• Training {self.model.__class__.__name__}...")
        self.model.fit(X, y)

        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        print("✓ Training complete")
        print(f"• CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        return self.model

    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        X_selected = X[self.kept_features]
        preds = self.model.predict(X_selected)
        if self.problem_type == "classification" and self.target_encoder:
            preds = self.target_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        """Get prediction probabilities (classification only)."""
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only for classification")
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        X_selected = X[self.kept_features]
        return self.model.predict_proba(X_selected)

    def save_model(self, filepath):
        """Save pipeline to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to: {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load pipeline from disk."""
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        print(f"✓ Model loaded from: {filepath}")
        return pipeline
