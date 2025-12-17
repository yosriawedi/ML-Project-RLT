# =========================
# model_RLT.py (COMPLETE + DSO2 OPTIMIZATION + PROGRESSIVE MUTING)
# =========================

# =========================
# IMPORTS
# =========================
import warnings
import pickle
from dataclasses import dataclass
from io import BytesIO
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, pearsonr

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from tabulate import tabulate

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)

from xgboost import XGBClassifier, XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

warnings.filterwarnings("ignore")


# =========================
# RLT FEATURE SELECTOR
# =========================
class RLTFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sélection de features basée sur ExtraTrees.feature_importances_.
    Entrée: numpy array (X) après preprocessing.
    Sortie: numpy array réduit.
    """
    def __init__(self, problem_type="classification", vi_threshold=0.01, min_features=5,
                 n_estimators=300, random_state=42, n_jobs=-1):
        self.problem_type = problem_type
        self.vi_threshold = vi_threshold
        self.min_features = min_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.support_ = None
        self.importances_ = None

    def fit(self, X, y):
        if self.problem_type == "classification":
            est = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            est = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

        est.fit(X, y)
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            raise ValueError("Estimator does not expose feature_importances_.")
        self.importances_ = np.asarray(importances)

        support = self.importances_ >= self.vi_threshold
        if support.sum() < self.min_features:
            top_idx = np.argsort(self.importances_)[::-1][:self.min_features]
            support = np.zeros_like(self.importances_, dtype=bool)
            support[top_idx] = True

        self.support_ = support
        return self

    def transform(self, X):
        if self.support_ is None:
            raise ValueError("RLTFeatureSelector not fitted.")
        return X[:, self.support_]


# =========================
# RLT BENCHMARK PIPELINE
# =========================
class RLTBenchmarkPipeline:
    def __init__(self, problem_type="classification", target_col=None,
                 test_size=0.2, random_state=42, vi_threshold=0.01):
        self.problem_type = problem_type.lower().strip()
        if self.problem_type not in ["classification", "regression"]:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.vi_threshold = vi_threshold

        self.preprocessor = None
        self.target_encoder = None
        self.feature_names_ = None

        self.fitted_models_ = {}
        self.best_model_name_ = None

        # Variable importance storage (after preprocessing)
        self.vi_scores_ = None
        self.high_vi_features_ = None
        self.low_vi_features_ = None
        self.vi_adaptive_threshold_ = None

        # DSO2: Optimized models storage
        self.fitted_models_optimized_ = {}
        self.best_model_optimized_name_ = None

        # DSO2: progressive muting curves (per embedded model)
        self.muting_curves_optimized_ = {}

    # -------------------
    # 1) Data Understanding
    # -------------------
    def data_understanding(self, df, target_col=None, max_desc_cols=15, show_plots=False):
        """
        EDA light. En mode déploiement/CI, laisser show_plots=False.
        """
        if target_col is None:
            target_col = self.target_col

        print("\n" + "=" * 70)
        print("DATA UNDERSTANDING (EDA LIGHT)")
        print("=" * 70)

        print(f"• Shape: {df.shape}")
        if target_col is not None and target_col in df.columns:
            print(f"• Target: {target_col}")
        else:
            print("• Target: (non fourni)")

        # Missing / duplicates
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        print(f"• Missing columns: {len(miss)}")
        if len(miss) > 0:
            print(miss.to_string())

        dups = int(df.duplicated().sum())
        print(f"• Duplicates: {dups}")

        # Types
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        print(f"• Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")

        # Target distribution / stats
        if target_col is not None and target_col in df.columns:
            y = df[target_col]
            if self.problem_type == "classification":
                print("• Target distribution (counts):")
                print(y.value_counts(dropna=False).to_string())
                print("• Target distribution (proportions):")
                print(y.value_counts(normalize=True, dropna=False).rename("proportion").to_string())
            else:
                print("• Target describe:")
                print(y.describe().to_string())

        # Feature stats
        if len(num_cols) > 0:
            print("• Numeric features describe (head):")
            print(df[num_cols].describe().T.head(max_desc_cols).to_string())

            # Boxplot + histogramme de la moyenne par ligne
            mean_values = df[num_cols].mean(axis=1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].boxplot(mean_values, vert=True)
            axes[0].set_title("Boxplot de la moyenne des features par ligne")
            axes[0].set_ylabel("Moyenne des features")

            axes[1].hist(mean_values, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            axes[1].set_title("Histogramme de la moyenne des features par ligne")
            axes[1].set_xlabel("Moyenne des features")
            axes[1].set_ylabel("Fréquence")

            fig.tight_layout()
            if show_plots:
                plt.show()
            plt.close(fig)

        # Correlation matrix (numeric)
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                annot=(len(num_cols) <= 15),
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=1,
                ax=ax
            )
            ax.set_title("Matrice de Corrélation")
            fig.tight_layout()
            if show_plots:
                plt.show()
            plt.close(fig)

            if target_col in num_cols:
                target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                print(f"\n🎯 Top 10 Corrélations avec {target_col}:")
                print(target_corr.head(10).to_string())

        print("\n✅ Data Understanding terminé!")

    # -------------------
    # 2) Build preprocessor
    # -------------------
    def _build_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = X.select_dtypes(exclude="number").columns.tolist()

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ],
            remainder="drop"
        )
        return preprocessor

    def _get_feature_names_out_safe(self):
        try:
            names = self.preprocessor.get_feature_names_out()
            return [str(n) for n in names]
        except Exception:
            return None

    # -------------------
    # 3) Preprocess
    # -------------------
    def preprocess(self, df, target_col=None, fit=True):
        if target_col is None:
            target_col = self.target_col
        if target_col is None:
            raise ValueError("target_col must be provided.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        if fit:
            self.preprocessor = self._build_preprocessor(X)
            X_proc = self.preprocessor.fit_transform(X)
            self.feature_names_ = self._get_feature_names_out_safe()
        else:
            if self.preprocessor is None:
                raise ValueError("Preprocessor not fitted. Call preprocess(..., fit=True) first.")
            X_proc = self.preprocessor.transform(X)

        if self.problem_type == "classification":
            if fit:
                self.target_encoder = None
                if y.dtype == "object" or str(y.dtype).startswith("category"):
                    self.target_encoder = LabelEncoder()
                    y_enc = self.target_encoder.fit_transform(y.astype(str))
                else:
                    y_enc = y.to_numpy()
            else:
                if self.target_encoder is not None:
                    y_enc = self.target_encoder.transform(y.astype(str))
                else:
                    y_enc = y.to_numpy()
        else:
            y_enc = y.to_numpy()

        return np.asarray(X_proc), np.asarray(y_enc)

    # -------------------
    # 4) Models dictionary
    # -------------------
    def _get_models(self):
        if self.problem_type == "classification":
            return {
                "RLT-ExtraTrees": Pipeline(steps=[
                    ("selector", RLTFeatureSelector(
                        problem_type="classification",
                        vi_threshold=self.vi_threshold,
                        min_features=5,
                        n_estimators=300,
                        random_state=self.random_state,
                        n_jobs=-1
                    )),
                    ("model", ExtraTreesClassifier(
                        n_estimators=300,
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ]),
                "RF": RandomForestClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1),
                "LASSO(LogReg)": LogisticRegression(
                    penalty="l1", solver="saga", C=10.0, max_iter=5000,
                    random_state=self.random_state
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.random_state, n_jobs=-1, verbosity=0
                ),
                "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=self.random_state),
            }
        else:
            return {
                "RLT-ExtraTrees": Pipeline(steps=[
                    ("selector", RLTFeatureSelector(
                        problem_type="regression",
                        vi_threshold=self.vi_threshold,
                        min_features=5,
                        n_estimators=300,
                        random_state=self.random_state,
                        n_jobs=-1
                    )),
                    ("model", ExtraTreesRegressor(
                        n_estimators=300,
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ]),
                "RF": RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
                "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
                "LASSO": Lasso(alpha=0.05, max_iter=8000, random_state=self.random_state),
                "XGBoost": XGBRegressor(
                    n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.random_state, n_jobs=-1, verbosity=0
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=400, learning_rate=0.05, random_state=self.random_state
                ),
            }

    # -------------------
    # 5) Metrics helpers
    # -------------------
    def _eval_test_metrics_from_preds(self, y_test, y_pred):
        if self.problem_type == "classification":
            return {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision_w": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "test_recall_w": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "test_f1_w": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            return {
                "test_r2": r2_score(y_test, y_pred),
                "test_rmse": rmse,
                "test_mae": mean_absolute_error(y_test, y_pred),
            }

    def _cv_scoring(self):
        if self.problem_type == "classification":
            return {"acc": "accuracy", "f1_w": "f1_weighted",
                    "prec_w": "precision_weighted", "rec_w": "recall_weighted"}
        else:
            return {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    # -------------------
    # 6) Variable importance
    # -------------------
    def _compute_rlt_variable_importance(self, X_train, y_train,
                                         vi_extra_weight=0.7, vi_stat_weight=0.3,
                                         min_features=5, plot=False, top_k_plot=20):
        """
        Variable importance sur données PRÉTRAITÉES (X_train numpy).
        - VI ExtraTrees: feature_importances_
        - VI statistiques: ANOVA (classification) / Pearson abs(corr) (regression)
        Stocke: self.vi_scores_, self.high_vi_features_, self.low_vi_features_, self.vi_adaptive_threshold_
        """
        print("\n" + "=" * 70)
        print("🧠 RLT: VARIABLE IMPORTANCE")
        print("=" * 70)
        print("📊 Calcul de Variable Importance...")

        tree_config = dict(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1
        )

        # 1) ExtraTrees VI
        if self.problem_type == "classification":
            et = ExtraTreesClassifier(**tree_config)
        else:
            et = ExtraTreesRegressor(**tree_config)

        et.fit(X_train, y_train)
        vi_et = np.asarray(et.feature_importances_, dtype=float)
        print("   ✅ Extra Trees VI calculé")

        # 2) Statistical VI
        n_features = X_train.shape[1]
        vi_stat = np.zeros(n_features, dtype=float)

        for j in range(n_features):
            try:
                col_vals = X_train[:, j]
                if self.problem_type == "classification":
                    labels = np.unique(y_train)
                    groups = [col_vals[y_train == label] for label in labels]
                    f_stat, _ = f_oneway(*groups)
                    vi_stat[j] = float(f_stat) / 1000.0
                else:
                    corr, _ = pearsonr(col_vals, y_train)
                    vi_stat[j] = abs(float(corr))
            except Exception:
                vi_stat[j] = 0.0

        print("   ✅ Statistical VI calculé")

        # 3) Normalize
        if vi_et.sum() > 0:
            vi_et = vi_et / vi_et.sum()
        if vi_stat.sum() > 0:
            vi_stat = vi_stat / vi_stat.sum()

        # 4) Aggregate
        vi_aggregate = vi_extra_weight * vi_et + vi_stat_weight * vi_stat

        # 5) DataFrame
        if self.feature_names_ is not None and len(self.feature_names_) == n_features:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        vi_df = pd.DataFrame({
            "Feature": feature_names,
            "VI_ExtraTrees": vi_et,
            "VI_Statistical": vi_stat,
            "VI_Aggregate": vi_aggregate
        }).sort_values("VI_Aggregate", ascending=False).reset_index(drop=True)

        self.vi_scores_ = vi_df

        print("\n🔝 Top 15 Features par VI:")
        print(vi_df.head(15).to_string(index=False))

        # Plot (optionnel)
        if plot:
            top = vi_df.head(int(top_k_plot)).iloc[::-1]  # pour barh (top en haut)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top["Feature"], top["VI_Aggregate"], color="steelblue")
            ax.set_xlabel("Variable Importance")
            ax.set_title(f"Top {min(top_k_plot, len(vi_df))} Features par VI")
            fig.tight_layout()
            plt.close(fig)

        # Muting adaptatif (diagnostic + stockage)
        print("\n🔇 Variable Muting (Seuil Adaptatif):")
        vi_values = vi_df["VI_Aggregate"].values
        vi_median = float(np.median(vi_values))
        vi_mean = float(np.mean(vi_values))
        vi_std = float(np.std(vi_values))

        adaptive_threshold = max(float(self.vi_threshold), vi_mean)

        print("   📊 Statistiques VI:")
        print(f"      - Médiane: {vi_median:.4f}")
        print(f"      - Moyenne: {vi_mean:.4f}")
        print(f"      - Écart-type: {vi_std:.4f}")
        print(f"   🎯 Seuil fixe (config): {self.vi_threshold}")
        print(f"   ⚡ Seuil adaptatif (utilisé): {adaptive_threshold:.4f}")

        high_mask = vi_df["VI_Aggregate"] >= adaptive_threshold
        high_feats = vi_df.loc[high_mask, "Feature"].tolist()
        low_feats = vi_df.loc[~high_mask, "Feature"].tolist()

        if len(high_feats) < min_features:
            print(f"   ⚠️  Seuil trop strict, gardons au moins {min_features} features")
            high_feats = vi_df.head(min_features)["Feature"].tolist()
            low_feats = vi_df.iloc[min_features:]["Feature"].tolist()
            adaptive_threshold = float(vi_df.iloc[min_features - 1]["VI_Aggregate"])

        self.high_vi_features_ = high_feats
        self.low_vi_features_ = low_feats
        self.vi_adaptive_threshold_ = adaptive_threshold

        n_total = n_features
        print("\n   ✂️  Résultat du Muting (diagnostic):")
        print(f"   - Original features: {n_total}")
        print(f"   - Features mutées: {len(low_feats)} ({len(low_feats) / n_total * 100:.1f}%)")
        print(f"   - Features gardées: {len(high_feats)} ({len(high_feats) / n_total * 100:.1f}%)")
        print(f"   - Seuil final: {adaptive_threshold:.4f}")

        print("\n✅ Variable Importance terminé!")
        return vi_df

    # -------------------
    # 6b) DSO2: Progressive muting evaluation (NEW)
    # -------------------
    def _progressive_muting_evaluation(
        self,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        vi_df: pd.DataFrame,
        base_model,
        min_features: int = 5,
        plot: bool = False,
        metric_plot: str | None = None
    ) -> pd.DataFrame:
        """
        Muting progressif:
        - Trier les features par VI croissante (moins importantes d'abord)
        - Enlever progressivement les moins importantes
        - Ré-entraîner (clone) et mesurer les performances à chaque étape
        """
        vi_sorted = vi_df.sort_values("VI_Aggregate", ascending=True).reset_index(drop=True)
        features_all = vi_sorted["Feature"].tolist()
        features_all = [f for f in features_all if f in X_train_df.columns]

        if len(features_all) == 0:
            raise ValueError("Aucune feature de vi_df n'existe dans X_train_df.")

        if min_features < 1:
            min_features = 1
        min_features = min(min_features, len(features_all))

        results = []
        max_i = max(0, len(features_all) - min_features)

        for i in range(max_i + 1):
            features_kept = features_all[i:]
            Xtr = X_train_df[features_kept]
            Xte = X_test_df[features_kept]

            model_i = clone(base_model)
            model_i.fit(Xtr, y_train)
            y_pred = model_i.predict(Xte)

            if self.problem_type == "classification":
                results.append({
                    "Step": i,
                    "Variables_Kept": len(features_kept),
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                })
            else:
                results.append({
                    "Step": i,
                    "Variables_Kept": len(features_kept),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred),
                })

        results_df = pd.DataFrame(results)

        if plot:
            if metric_plot is None:
                metric_plot = "Accuracy" if self.problem_type == "classification" else "R2"

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df["Step"], results_df[metric_plot], marker="o", color="darkorange")
            ax.set_title(f"Evolution of {metric_plot} with muting progressif")
            ax.set_xlabel("Etape (nombre de variables mutées)")
            ax.set_ylabel(metric_plot)
            ax.grid(True)
            plt.show()
            plt.close(fig)

        return results_df

    # -------------------
    # 7) Benchmark
    # -------------------
    def benchmark(self, df, target_col=None, do_cv=True, cv_folds=5, compute_vi=True):
        if target_col is None:
            target_col = self.target_col
        if target_col is None:
            raise ValueError("target_col must be provided.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in dataframe.")

        strat = df[target_col] if self.problem_type == "classification" else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )

        X_train, y_train = self.preprocess(train_df, target_col=target_col, fit=True)
        X_test, y_test = self.preprocess(test_df, target_col=target_col, fit=False)

        # Variable importance (optionnel)
        if compute_vi:
            try:
                self._compute_rlt_variable_importance(X_train, y_train, plot=False)
            except Exception as e:
                print("\n⚠️  Variable Importance non calculée:", str(e))

        models = self._get_models()
        rows = []

        for name, model in models.items():
            # Fit once
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            row = {"model": name}
            row.update(self._eval_test_metrics_from_preds(y_test, y_pred))

            # Store fitted model
            self.fitted_models_[name] = model

            # CV
            if do_cv:
                scoring = self._cv_scoring()
                cv = cross_validate(
                    model, X_train, y_train,
                    scoring=scoring,
                    cv=cv_folds,
                    n_jobs=-1,
                    return_train_score=False
                )
                for k, v in cv.items():
                    if k.startswith("test_"):
                        key = "cv_" + k.replace("test_", "")
                        row[key] = float(np.mean(v))

                # Fix negatives for regression
                if self.problem_type == "regression":
                    if "cv_rmse" in row:
                        row["cv_rmse"] = -row["cv_rmse"]
                    if "cv_mae" in row:
                        row["cv_mae"] = -row["cv_mae"]

            rows.append(row)

        results = pd.DataFrame(rows)
        sort_key = "test_accuracy" if self.problem_type == "classification" else "test_r2"
        results = results.sort_values(sort_key, ascending=False).reset_index(drop=True)
        self.best_model_name_ = results.loc[0, "model"] if len(results) else None

        # Features kept for RLT selector (post-preprocessing)
        if "RLT-ExtraTrees" in self.fitted_models_ and self.feature_names_ is not None:
            rlt_model = self.fitted_models_["RLT-ExtraTrees"]
            selector = rlt_model.named_steps.get("selector", None)
            if selector is not None and getattr(selector, "support_", None) is not None:
                support = selector.support_
                kept = [fn for fn, keep in zip(self.feature_names_, support) if keep]

                print("\n" + "=" * 70)
                print("RLT: FEATURES KEPT (after preprocessing)")
                print("=" * 70)
                print(f"• Kept features: {len(kept)} / {len(self.feature_names_)}")
                print("• Top kept (first 30):")
                for f in kept[:30]:
                    print("  -", f)

        return results

    # -------------------
    # 7b) DSO2: OPTIMIZED BENCHMARK WITH FEATURE ENGINEERING (+ progressive muting)
    # -------------------
    def benchmark_optimized(self, df, target_col=None, top_n=15, plot=False,
                            progressive_muting=False, muting_min_features=5, muting_metric_plot=None):
        """
        🔵 OPTIMISATION RLT: Feature Engineering + Embedded Models

        Ajout (optionnel):
        - progressive_muting=True : calcule la courbe de performance en retirant
          progressivement les features les moins importantes (selon VI).
        """
        if target_col is None:
            target_col = self.target_col
        if target_col is None:
            raise ValueError("target_col must be provided.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in dataframe.")

        print("\n" + "=" * 70)
        print("🔵 OPTIMISATION RLT: Feature Engineering + Embedded Models")
        print("=" * 70)

        # Split train/test
        strat = df[target_col] if self.problem_type == "classification" else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )

        X_train, y_train = self.preprocess(train_df, target_col=target_col, fit=True)
        X_test, y_test = self.preprocess(test_df, target_col=target_col, fit=False)

        # Convertir en DataFrame pour faciliter les opérations
        if self.feature_names_ is not None:
            X_train_df = pd.DataFrame(X_train, columns=self.feature_names_)
            X_test_df = pd.DataFrame(X_test, columns=self.feature_names_)
        else:
            X_train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(X_train.shape[1])])
            X_test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])

        # Configuration des modèles
        tree_config = dict(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1
        )

        boosting_config = dict(
            n_estimators=400,
            learning_rate=0.05,
            random_state=self.random_state
        )

        xgboost_config = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )

        # Définir les modèles embedded
        if self.problem_type == "classification":
            embedded_models = {
                "ExtraTrees": ExtraTreesClassifier(**tree_config),
                "RandomForest": RandomForestClassifier(**tree_config),
                "GradientBoosting": GradientBoostingClassifier(**boosting_config),
                "XGBoost": XGBClassifier(**xgboost_config),
            }
        else:
            embedded_models = {
                "ExtraTrees": ExtraTreesRegressor(**tree_config),
                "RandomForest": RandomForestRegressor(**tree_config),
                "GradientBoosting": GradientBoostingRegressor(**boosting_config),
                "XGBoost": XGBRegressor(**xgboost_config),
            }

        # Fonction pour créer des combinaisons linéaires
        def create_additional_combinations(X: pd.DataFrame, vi_scores: pd.DataFrame, top_n: int = 15):
            X_comb = X.copy()
            top_features = vi_scores.head(top_n)['Feature'].tolist()
            combinations_created = 0

            for i in range(len(top_features) - 1):
                for j in range(i + 1, len(top_features)):
                    feat1 = top_features[i]
                    feat2 = top_features[j]

                    if feat1 not in X.columns or feat2 not in X.columns:
                        continue

                    w1 = vi_scores.loc[vi_scores['Feature'] == feat1, 'VI_Aggregate'].values[0]
                    w2 = vi_scores.loc[vi_scores['Feature'] == feat2, 'VI_Aggregate'].values[0]
                    total = w1 + w2
                    w1_norm = w1 / total if total > 0 else 0.5
                    w2_norm = w2 / total if total > 0 else 0.5
                    new_col = f"LC_opt_{i}_{j}"
                    X_comb[new_col] = w1_norm * X[feat1] + w2_norm * X[feat2]
                    combinations_created += 1

            print(f"   ✅ {combinations_created} nouvelles combinaisons créées")
            return X_comb

        # Exécution pour chaque modèle embedded
        results_opt = []

        for name, model in embedded_models.items():
            print(f"\n⚙️  RLT Optimization using embedded model: {name} ...")

            # Calcul VI (sur X_train numpy)
            vi_df = self._compute_rlt_variable_importance(X_train, y_train, plot=False)

            # 🔇 DSO2: Muting progressif (optionnel) — sur toutes les features prétraitées
            if progressive_muting:
                try:
                    muting_df = self._progressive_muting_evaluation(
                        X_train_df=X_train_df,
                        X_test_df=X_test_df,
                        y_train=y_train,
                        y_test=y_test,
                        vi_df=vi_df,
                        base_model=model,
                        min_features=muting_min_features,
                        plot=plot,
                        metric_plot=muting_metric_plot
                    )
                    self.muting_curves_optimized_[name] = muting_df
                except Exception as e:
                    print(f"   ⚠️ Muting progressif ignoré pour {name}: {e}")

            # Muting adaptatif (comme ton code initial)
            vi_values = vi_df["VI_Aggregate"].values
            adaptive_threshold = max(np.mean(vi_values), self.vi_threshold)
            high_vi_features = vi_df[vi_df["VI_Aggregate"] >= adaptive_threshold]["Feature"].tolist()

            high_vi_features = [f for f in high_vi_features if f in X_train_df.columns]
            if len(high_vi_features) == 0:
                high_vi_features = X_train_df.columns.tolist()

            X_train_mut = X_train_df[high_vi_features]
            X_test_mut = X_test_df[high_vi_features]

            # Ajouter combinaisons linéaires
            X_train_rlt_opt = create_additional_combinations(X_train_mut, vi_df, top_n=top_n)
            X_test_rlt_opt = create_additional_combinations(X_test_mut, vi_df, top_n=top_n)

            # Entraînement final (DSO2)
            model.fit(X_train_rlt_opt, y_train)
            y_pred = model.predict(X_test_rlt_opt)

            # Stockage du modèle
            self.fitted_models_optimized_[name] = model

            # Calcul metrics
            if self.problem_type == "classification":
                results_opt.append({
                    "model": name,
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "test_f1_w": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "test_precision_w": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "test_recall_w": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "features_used": X_train_rlt_opt.shape[1]
                })
            else:
                results_opt.append({
                    "model": name,
                    "test_r2": r2_score(y_test, y_pred),
                    "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "test_mae": mean_absolute_error(y_test, y_pred),
                    "features_used": X_train_rlt_opt.shape[1]
                })

        # Résultats optimisés
        results_opt_df = pd.DataFrame(results_opt)
        sort_key = "test_accuracy" if self.problem_type == "classification" else "test_r2"
        results_opt_df = results_opt_df.sort_values(sort_key, ascending=False).reset_index(drop=True)
        self.best_model_optimized_name_ = results_opt_df.loc[0, "model"] if len(results_opt_df) else None

        print("\n📋 Résultats Optimisés RLT:")
        print(results_opt_df.to_string(index=False))

        # Comparaison visuelle (barplot)
        if plot:
            metric_to_plot = "test_accuracy" if self.problem_type == "classification" else "test_r2"
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(results_opt_df["model"], results_opt_df[metric_to_plot], color='darkorange')
            ax.set_title(f"RLT Optimisé: Comparaison des modèles — Metric: {metric_to_plot}")
            ax.set_ylabel(metric_to_plot)
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

        return results_opt_df

    # -------------------
    # 8) Predict
    # -------------------
    def predict(self, df, model_name=None):
        if model_name is None:
            model_name = self.best_model_name_
        if model_name is None or model_name not in self.fitted_models_:
            raise ValueError("No fitted model found. Run benchmark() first or provide a valid model_name.")

        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Run benchmark() first.")

        X = df.copy()
        if self.target_col is not None and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        X_proc = self.preprocessor.transform(X)
        preds = self.fitted_models_[model_name].predict(X_proc)

        if self.problem_type == "classification" and self.target_encoder is not None:
            preds = self.target_encoder.inverse_transform(preds.astype(int))

        return preds

    def predict_proba(self, df, model_name=None):
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification.")
        if model_name is None:
            model_name = self.best_model_name_
        if model_name is None or model_name not in self.fitted_models_:
            raise ValueError("No fitted model found. Run benchmark() first or provide a valid model_name.")

        model = self.fitted_models_[model_name]
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Model '{model_name}' does not support predict_proba().")

        X = df.copy()
        if self.target_col is not None and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        X_proc = self.preprocessor.transform(X)
        return model.predict_proba(X_proc)

    # -------------------
    # 9) Persistence
    # -------------------
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Pipeline saved to: {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"✓ Pipeline loaded from: {filepath}")
        return obj
def compute_vi_xai(X, y, problem_type, random_state=42, n_jobs=-1):
    """
    Variable importance for XAI (lightweight version)
    """
    if problem_type == 'classification':
        model = ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs
        )
    
    model.fit(X, y)
    vi = model.feature_importances_
    return vi / vi.sum() if vi.sum() > 0 else vi


def linear_combinations_xai(X, vi, n_comb=2):
    """
    Simple linear combinations for XAI
    """
    if X.shape[1] < 2:
        return X
    
    top_idx = np.argsort(vi)[-min(10, X.shape[1]):]
    X_new = X.copy()
    
    for i in range(len(top_idx) - 1):
        w1 = vi[top_idx[i]]
        w2 = vi[top_idx[i + 1]]
        total = w1 + w2 if (w1 + w2) > 0 else 1.0
        new_feat = (w1 / total) * X[:, top_idx[i]] + (w2 / total) * X[:, top_idx[i + 1]]
        X_new = np.column_stack([X_new, new_feat])
        
        if X_new.shape[1] >= X.shape[1] + n_comb:
            break
    
    return X_new


# =========================
# XAI: RLT Feature Space Builder
# =========================

def build_rlt_feature_space(X_train, y_train, X_test, vi_threshold=0.01, 
                           problem_type='classification', random_state=42):
    """
    Build RLT feature space with variable importance filtering + linear combinations
    
    Returns:
        X_train_rlt, X_test_rlt, vi_selected, selected_indices
    """
    # Compute VI
    vi = compute_vi_xai(X_train, y_train, problem_type, random_state)
    
    # Adaptive threshold
    threshold = max(vi_threshold, np.mean(vi))
    selected = np.where(vi >= threshold)[0]
    
    # Keep at least 5 features
    if len(selected) < 5:
        selected = np.argsort(vi)[-5:]
    
    # Select features
    X_train_rlt = X_train[:, selected]
    X_test_rlt = X_test[:, selected]
    vi_sel = vi[selected]
    
    # Add linear combinations
    X_train_rlt = linear_combinations_xai(X_train_rlt, vi_sel, n_comb=2)
    X_test_rlt = linear_combinations_xai(X_test_rlt, vi_sel, n_comb=2)
    
    return X_train_rlt, X_test_rlt, vi_sel, selected


# =========================
# XAI: RLT Heatmap
# =========================

def generate_rlt_heatmap(rlt_model, figsize=(12, 2)):
    """
    Generate RLT Feature Heatmap (Intrinsic Explanation)
    
    Args:
        rlt_model: trained model with feature_importances_
        
    Returns:
        BytesIO buffer containing PNG image
    """
    feature_scores = rlt_model.feature_importances_
    feature_scores = feature_scores / feature_scores.sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        feature_scores.reshape(1, -1),
        cmap="coolwarm",
        cbar=True,
        ax=ax
    )
    ax.set_title("RLT Feature Heatmap (Intrinsic Explanation)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Selected / Constructed Features")
    ax.set_yticks([])
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    
    return buf


# =========================
# XAI: SHAP Explanation
# =========================

def generate_shap_explanation(rlt_model, X_test_rlt, instance_idx=0, plot_type="bar"):
    """
    Generate SHAP explanation for a single instance
    
    Args:
        rlt_model: trained model
        X_test_rlt: test data (numpy array)
        instance_idx: which test instance to explain
        plot_type: "bar", "waterfall", or "force"
        
    Returns:
        BytesIO buffer containing PNG image, or dict with SHAP values
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    X_test_np = np.asarray(X_test_rlt)
    x_instance = X_test_np[instance_idx:instance_idx+1]
    
    # SHAP explainer
    explainer = shap.Explainer(rlt_model, X_test_np)
    shap_values = explainer(x_instance)
    
    # Generate plot
    fig = plt.figure(figsize=(12, 6))
    
    if plot_type == "bar":
        shap.plots.bar(shap_values[0], show=False)
    elif plot_type == "waterfall":
        shap.plots.waterfall(shap_values[0], show=False)
    elif plot_type == "force":
        shap.plots.force(shap_values[0], show=False, matplotlib=True)
    else:
        shap.plots.bar(shap_values[0], show=False)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    
    return buf, {
        "shap_values": shap_values.values[0].tolist(),
        "base_value": float(shap_values.base_values[0]) if hasattr(shap_values.base_values, '__iter__') else float(shap_values.base_values),
        "feature_values": x_instance[0].tolist()
    }


# =========================
# XAI: Comparison Plot (RLT vs Standard)
# =========================

def generate_comparison_plot(rlt_model, standard_model, X_test_rlt, X_test_standard, 
                            y_test, problem_type='classification'):
    """
    Compare RLT model vs Standard model predictions
    
    Returns:
        BytesIO buffer containing PNG image
    """
    # Predictions
    y_pred_rlt = rlt_model.predict(X_test_rlt)
    y_pred_std = standard_model.predict(X_test_standard)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Predictions comparison
    n_samples = min(50, len(y_test))
    x_axis = np.arange(n_samples)
    
    axes[0].plot(x_axis, y_test[:n_samples], 'o-', label='True', alpha=0.7, markersize=8)
    axes[0].plot(x_axis, y_pred_rlt[:n_samples], 's-', label='RLT', alpha=0.7, markersize=6)
    axes[0].plot(x_axis, y_pred_std[:n_samples], '^-', label='Standard', alpha=0.7, markersize=6)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Prediction')
    axes[0].set_title('Predictions Comparison (First 50 samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature importance comparison
    rlt_fi = rlt_model.feature_importances_
    std_fi = standard_model.feature_importances_
    
    x = np.arange(min(len(rlt_fi), 20))
    width = 0.35
    
    axes[1].bar(x - width/2, rlt_fi[:len(x)], width, label='RLT', alpha=0.8)
    axes[1].bar(x + width/2, std_fi[:len(x)], width, label='Standard', alpha=0.8)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Importance')
    axes[1].set_title('Feature Importance Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    error_rlt = np.abs(y_test - y_pred_rlt)
    error_std = np.abs(y_test - y_pred_std)
    
    axes[2].hist(error_rlt, bins=30, alpha=0.6, label='RLT', edgecolor='black')
    axes[2].hist(error_std, bins=30, alpha=0.6, label='Standard', edgecolor='black')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    
    return buf


# =========================
# XAI: Complete Pipeline
# =========================

class RLTXAIExplainer:
    """
    Complete XAI pipeline for RLT models
    """
    
    def __init__(self, problem_type='classification', vi_threshold=0.01, random_state=42):
        self.problem_type = problem_type
        self.vi_threshold = vi_threshold
        self.random_state = random_state
        
        self.X_train_rlt = None
        self.X_test_rlt = None
        self.rlt_model = None
        self.vi_selected = None
        self.selected_indices = None
        
    def fit(self, X_train, y_train, X_test):
        """
        Build RLT feature space and train XAI model
        """
        # Build RLT features
        self.X_train_rlt, self.X_test_rlt, self.vi_selected, self.selected_indices = \
            build_rlt_feature_space(
                X_train, y_train, X_test,
                vi_threshold=self.vi_threshold,
                problem_type=self.problem_type,
                random_state=self.random_state
            )
        
        # Train RLT model
        if self.problem_type == 'classification':
            self.rlt_model = ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.rlt_model = ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        self.rlt_model.fit(self.X_train_rlt, y_train)
        
        return self
    
    def get_heatmap(self):
        """Generate RLT heatmap"""
        if self.rlt_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return generate_rlt_heatmap(self.rlt_model)
    
    def get_shap_explanation(self, instance_idx=0, plot_type="bar"):
        """Generate SHAP explanation"""
        if self.rlt_model is None or self.X_test_rlt is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return generate_shap_explanation(
            self.rlt_model, 
            self.X_test_rlt, 
            instance_idx, 
            plot_type
        )
    
    def get_comparison(self, standard_model, X_test_standard, y_test):
        """Generate comparison plot"""
        if self.rlt_model is None or self.X_test_rlt is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return generate_comparison_plot(
            self.rlt_model,
            standard_model,
            self.X_test_rlt,
            X_test_standard,
            y_test,
            self.problem_type
        )
    
     
   

