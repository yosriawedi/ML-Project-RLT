from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shap
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import os

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)

# --- Viz libs (server-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---- Ton modèle ----
# (AJOUT) RLTXAIExplainer + SHAP_AVAILABLE
from model_RLT import RLTBenchmarkPipeline, RLTXAIExplainer, SHAP_AVAILABLE


# =========================
# Global state
# =========================
MODEL_PATH = "last_pipeline.pkl"

pipeline: Optional[RLTBenchmarkPipeline] = None
CURRENT_DF: Optional[pd.DataFrame] = None
CURRENT_TARGET: Optional[str] = None
CURRENT_PROBLEM: Optional[str] = None
CURRENT_FILENAME: Optional[str] = None

BENCHMARK_RESULTS_DSO1: Optional[pd.DataFrame] = None  # DSO1
BENCHMARK_RESULTS_DSO2: Optional[pd.DataFrame] = None  # DSO2

# NEW: courbes muting progressif (DSO2)
MUTING_CURVES_DSO2: Optional[Dict[str, pd.DataFrame]] = None

# (AJOUT) XAI globals
XAI_READY: bool = False
XAI_EXPLAINER: Optional[RLTXAIExplainer] = None

if os.path.exists(MODEL_PATH):
    pipeline = RLTBenchmarkPipeline.load(MODEL_PATH)


# =========================
# App
# =========================
app = FastAPI(
    title="Generic RLT Benchmark Pipeline API",
    description="Upload dataset -> train/benchmark (DSO1 + DSO2) -> predict + EDA dashboard.",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================
# Schemas
# =========================
class PredictRow(BaseModel):
    features: Dict[str, Any]
    model_name: Optional[str] = None
    use_optimized: Optional[bool] = False


# =========================
# Utils
# =========================
def _ensure_pipeline_loaded():
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Aucun modèle chargé. Uploade un dataset et entraîne d'abord.")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _fig_to_png_response(fig) -> StreamingResponse:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def _auto_problem_type(y: pd.Series) -> str:
    """Détecte automatiquement le type de problème ML"""
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 15:
        return "classification"
    return "regression"


def _get_current_df() -> pd.DataFrame:
    global CURRENT_DF
    if CURRENT_DF is None:
        raise HTTPException(status_code=400, detail="Aucun dataset en mémoire. Uploade un CSV d'abord.")
    return CURRENT_DF.copy()


def _get_target_col(df: pd.DataFrame) -> str:
    global CURRENT_TARGET
    if CURRENT_TARGET is not None and CURRENT_TARGET in df.columns:
        return CURRENT_TARGET
    return df.columns[-1]


# =========================
# TRAIN / UPLOAD - DSO1 + DSO2 (+ muting progressif DSO2)
# =========================
@app.post("/upload-and-train")
async def upload_and_train(
    file: UploadFile = File(...),
    problem_type: Optional[str] = Query(default=None, description="'regression' ou 'classification'. Si None, détection automatique."),
    top_n: Optional[int] = Query(default=15, description="Nombre de top features pour DSO2"),
    progressive_muting: bool = Query(default=True, description="DSO2: calcule la courbe de muting progressif"),
    muting_min_features: int = Query(default=5, description="DSO2 muting: garder au moins N features"),
):
    """
    Upload un fichier CSV et entraîne le pipeline avec DSO1 et DSO2.
    """
    global pipeline, CURRENT_DF, CURRENT_TARGET, CURRENT_PROBLEM, CURRENT_FILENAME
    global BENCHMARK_RESULTS_DSO1, BENCHMARK_RESULTS_DSO2, MUTING_CURVES_DSO2
    global XAI_READY, XAI_EXPLAINER  # (AJOUT)

    DEFAULT_VI_THRESHOLD = 0.01
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_SEED = 42
    DEFAULT_DO_CV = False
    DEFAULT_CV_FOLDS = 5

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV invalide: il faut au moins 1 feature + 1 target.")

        target_col = df.columns[-1]
        df = df.dropna(subset=[target_col])

        y = df[target_col]
        if problem_type is None:
            ptype = _auto_problem_type(y)
        else:
            ptype = problem_type.lower().strip()
            if ptype not in ["regression", "classification"]:
                raise HTTPException(status_code=400, detail="problem_type doit être 'regression' ou 'classification'.")

        CURRENT_DF = df.copy()
        CURRENT_TARGET = target_col
        CURRENT_PROBLEM = ptype
        CURRENT_FILENAME = file.filename

        pipeline = RLTBenchmarkPipeline(
            problem_type=ptype,
            target_col=target_col,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_SEED,
            vi_threshold=DEFAULT_VI_THRESHOLD,
        )

        # ========================================
        # DSO1: BENCHMARK STANDARD
        # ========================================
        print("\n" + "=" * 80)
        print("DSO1: BENCHMARK STANDARD")
        print("=" * 80)

        results_dso1 = pipeline.benchmark(
            df,
            target_col=target_col,
            do_cv=DEFAULT_DO_CV,
            cv_folds=DEFAULT_CV_FOLDS
        )

        BENCHMARK_RESULTS_DSO1 = results_dso1.copy()
        all_models_dso1 = results_dso1.to_dict(orient="records")

        preds_dso1 = pipeline.predict(df)
        y_true = df[target_col].to_numpy()

        if ptype == "regression":
            metrics_dso1 = {
                "r2": float(r2_score(y_true, preds_dso1)),
                "rmse": _rmse(y_true, preds_dso1),
                "mae": float(mean_absolute_error(y_true, preds_dso1)),
            }
        else:
            metrics_dso1 = {
                "accuracy": float(accuracy_score(y_true, preds_dso1)),
                "f1_weighted": float(f1_score(y_true, preds_dso1, average="weighted")),
            }

        # ========================================
        # DSO2: BENCHMARK OPTIMISÉ (+ muting progressif)
        # ========================================
        print("\n" + "=" * 80)
        print("DSO2: BENCHMARK OPTIMISÉ")
        print("=" * 80)

        results_dso2 = pipeline.benchmark_optimized(
            df,
            target_col=target_col,
            top_n=top_n,
            plot=False,
            progressive_muting=progressive_muting,
            muting_min_features=muting_min_features,
            muting_metric_plot=None
        )

        BENCHMARK_RESULTS_DSO2 = results_dso2.copy()
        all_models_dso2 = results_dso2.to_dict(orient="records")

        # ✅ CORRECTION: Stockage courbes muting
        print("\n" + "=" * 80)
        print("🔍 TRANSFERT DES COURBES MUTING VERS VARIABLE GLOBALE")
        print("=" * 80)

        MUTING_CURVES_DSO2 = {}
        if hasattr(pipeline, "muting_curves_optimized_") and pipeline.muting_curves_optimized_:
            print(f"✅ {len(pipeline.muting_curves_optimized_)} courbes trouvées dans pipeline")
            for mname, curve_df in pipeline.muting_curves_optimized_.items():
                MUTING_CURVES_DSO2[mname] = curve_df.copy()
                print(f"   - {mname}: {len(curve_df)} lignes copiées")
        else:
            print("❌ Aucune courbe de muting dans pipeline!")
            print(f"   Attribut existe? {hasattr(pipeline, 'muting_curves_optimized_')}")
            if hasattr(pipeline, "muting_curves_optimized_"):
                print(f"   Valeur: {pipeline.muting_curves_optimized_}")

        print(f"📊 MUTING_CURVES_DSO2 final: {len(MUTING_CURVES_DSO2)} modèles")
        print("=" * 80 + "\n")

        # ========================================
        # (AJOUT) XAI: Initialisation (Heatmap / SHAP / Comparison)
        # ========================================
        try:
            from sklearn.model_selection import train_test_split  # (AJOUT local)

            strat = df[target_col] if ptype == "classification" else None
            train_xai_df, test_xai_df = train_test_split(
                df,
                test_size=DEFAULT_TEST_SIZE,
                random_state=DEFAULT_SEED,
                stratify=strat
            )
            X_train_xai, y_train_xai = pipeline.preprocess(train_xai_df, target_col=target_col, fit=True)
            X_test_xai, _y_test_xai = pipeline.preprocess(test_xai_df, target_col=target_col, fit=False)

            XAI_EXPLAINER = RLTXAIExplainer(
                problem_type=ptype,
                vi_threshold=DEFAULT_VI_THRESHOLD,
                random_state=DEFAULT_SEED
            ).fit(X_train_xai, y_train_xai, X_test_xai)

            XAI_READY = True
        except Exception as _xai_e:
            XAI_READY = False
            XAI_EXPLAINER = None
            print(f"⚠️ XAI init failed: {_xai_e}")

        metric_key = "test_accuracy" if ptype == "classification" else "test_r2"
        best_dso2_score = float(results_dso2[metric_key].max()) if len(results_dso2) else float("nan")

        if len(results_dso2) > 0:
            if ptype == "regression":
                best_dso2_metrics = results_dso2.iloc[0][["test_r2", "test_rmse", "test_mae"]].to_dict()
                metrics_dso2 = {
                    "r2": float(best_dso2_metrics["test_r2"]),
                    "rmse": float(best_dso2_metrics["test_rmse"]),
                    "mae": float(best_dso2_metrics["test_mae"]),
                }
            else:
                best_dso2_metrics = results_dso2.iloc[0][["test_accuracy", "test_f1_w"]].to_dict()
                metrics_dso2 = {
                    "accuracy": float(best_dso2_metrics["test_accuracy"]),
                    "f1_weighted": float(best_dso2_metrics["test_f1_w"]),
                }
        else:
            metrics_dso2 = {}

        # ========================================
        # COMPARAISON DSO1 vs DSO2
        # ========================================
        best_dso1_score = float(results_dso1[metric_key].max()) if len(results_dso1) else float("nan")
        if np.isfinite(best_dso1_score) and best_dso1_score != 0 and np.isfinite(best_dso2_score):
            improvement = ((best_dso2_score - best_dso1_score) / abs(best_dso1_score)) * 100
        else:
            improvement = float("nan")

        pipeline.save(MODEL_PATH)

        return {
            "status": "success",
            "message": "Dataset uploadé, pipeline entraîné (DSO1 + DSO2) et sauvegardé.",
            "filename": CURRENT_FILENAME,
            "problem_type": ptype,
            "problem_type_detection": "auto" if problem_type is None else "manuel",
            "target_col": target_col,
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,

            "dso1": {
                "best_model": pipeline.best_model_name_,
                "metrics_full_data": metrics_dso1,
                "all_models_benchmark": all_models_dso1,
                "n_models_tested": len(results_dso1),
                "top_model": results_dso1.iloc[0].to_dict() if len(results_dso1) > 0 else None,
                "all_models": [row["model"] for row in all_models_dso1],
            },

            "dso2": {
                "best_model": pipeline.best_model_optimized_name_,
                "metrics_best_model": metrics_dso2,
                "all_models_benchmark": all_models_dso2,
                "n_models_tested": len(results_dso2),
                "top_model": results_dso2.iloc[0].to_dict() if len(results_dso2) > 0 else None,
                "all_models": [row["model"] for row in all_models_dso2],
                "muting": {
                    "enabled": bool(progressive_muting),
                    "available": bool(MUTING_CURVES_DSO2) and (len(MUTING_CURVES_DSO2) > 0),
                    "models": list(MUTING_CURVES_DSO2.keys()) if MUTING_CURVES_DSO2 else [],
                    "results_url": "/dso2/muting-results",
                    "plot_url_example": "/dso2/muting-plot.png?model_name=RandomForest",
                    # ✅ AJOUT CRUCIAL : best_model_data
                    "best_model_data": (
                        {
                            "model_name": pipeline.best_model_optimized_name_,
                            "columns": MUTING_CURVES_DSO2[pipeline.best_model_optimized_name_].columns.tolist(),
                            "data": MUTING_CURVES_DSO2[pipeline.best_model_optimized_name_].to_dict(orient="records")
                        }
                        if MUTING_CURVES_DSO2
                           and pipeline.best_model_optimized_name_
                           and pipeline.best_model_optimized_name_ in MUTING_CURVES_DSO2
                        else None
                    )
                }
            },

            "comparison": {
                "metric_used": metric_key,
                "dso1_best_score": best_dso1_score,
                "dso2_best_score": best_dso2_score,
                "improvement_percent": float(improvement) if np.isfinite(improvement) else None,
                "winner": "DSO2" if (np.isfinite(best_dso1_score) and np.isfinite(best_dso2_score) and best_dso2_score > best_dso1_score) else "DSO1",
            },

            "model_path": MODEL_PATH,
            "eda_dashboard_url": "/eda/dashboard",
            "benchmark_results_url": "/benchmark-results",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement : {str(e)}")


# =========================
# ENDPOINT: BENCHMARK RESULTS (DSO1 + DSO2)
# =========================
@app.get("/benchmark-results")
def get_benchmark_results():
    global BENCHMARK_RESULTS_DSO1, BENCHMARK_RESULTS_DSO2, MUTING_CURVES_DSO2

    _ensure_pipeline_loaded()

    if BENCHMARK_RESULTS_DSO1 is None and BENCHMARK_RESULTS_DSO2 is None:
        raise HTTPException(
            status_code=400,
            detail="Aucun résultat de benchmark disponible. Entraînez d'abord le pipeline via /upload-and-train."
        )

    response = {
        "status": "success",
        "problem_type": pipeline.problem_type,
        "target_col": pipeline.target_col,
    }

    if BENCHMARK_RESULTS_DSO1 is not None:
        response["dso1"] = {
            "best_model": pipeline.best_model_name_,
            "n_models_tested": len(BENCHMARK_RESULTS_DSO1),
            "models_tested": BENCHMARK_RESULTS_DSO1["model"].tolist(),
            "detailed_results": BENCHMARK_RESULTS_DSO1.to_dict(orient="records"),
            "results_dataframe": {
                "columns": BENCHMARK_RESULTS_DSO1.columns.tolist(),
                "data": BENCHMARK_RESULTS_DSO1.values.tolist()
            }
        }

    if BENCHMARK_RESULTS_DSO2 is not None:
        dso2_response = {
            "best_model": pipeline.best_model_optimized_name_,
            "n_models_tested": len(BENCHMARK_RESULTS_DSO2),
            "models_tested": BENCHMARK_RESULTS_DSO2["model"].tolist(),
            "detailed_results": BENCHMARK_RESULTS_DSO2.to_dict(orient="records"),
            "results_dataframe": {
                "columns": BENCHMARK_RESULTS_DSO2.columns.tolist(),
                "data": BENCHMARK_RESULTS_DSO2.values.tolist()
            },
            # ✅ Informations sur le muting
            "muting": {
                "available": bool(MUTING_CURVES_DSO2) and (len(MUTING_CURVES_DSO2) > 0),
                "models": list(MUTING_CURVES_DSO2.keys()) if MUTING_CURVES_DSO2 else [],
                "results_url": "/dso2/muting-results",
                "plot_url_example": "/dso2/muting-plot.png?model_name=RandomForest",
                # ✅ AJOUT CRUCIAL : best_model_data
                "best_model_data": (
                    {
                        "model_name": pipeline.best_model_optimized_name_,
                        "columns": MUTING_CURVES_DSO2[pipeline.best_model_optimized_name_].columns.tolist(),
                        "data": MUTING_CURVES_DSO2[pipeline.best_model_optimized_name_].to_dict(orient="records")
                    }
                    if MUTING_CURVES_DSO2
                       and pipeline.best_model_optimized_name_
                       and pipeline.best_model_optimized_name_ in MUTING_CURVES_DSO2
                    else None
                )
            }
        }

        response["dso2"] = dso2_response

    return response


# =========================
# NEW: DSO2 MUTING RESULTS (JSON)
# =========================
@app.get("/dso2/muting-results")
def get_dso2_muting_results(model_name: Optional[str] = Query(default=None)):
    _ensure_pipeline_loaded()

    if MUTING_CURVES_DSO2 is None or len(MUTING_CURVES_DSO2) == 0:
        raise HTTPException(status_code=400, detail="Aucune courbe de muting disponible. Active progressive_muting lors du train.")

    if model_name is not None:
        if model_name not in MUTING_CURVES_DSO2:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in muting curves.")
        df_curve = MUTING_CURVES_DSO2[model_name]
        return {
            "status": "success",
            "model_name": model_name,
            "columns": df_curve.columns.tolist(),
            "data": df_curve.to_dict(orient="records")
        }

    return {
        "status": "success",
        "models": list(MUTING_CURVES_DSO2.keys()),
        "curves": {
            m: {
                "columns": df_curve.columns.tolist(),
                "data": df_curve.to_dict(orient="records")
            }
            for m, df_curve in MUTING_CURVES_DSO2.items()
        }
    }


# =========================
# NEW: DSO2 MUTING PLOT (PNG)
# =========================
@app.get("/dso2/muting-plot.png")
def dso2_muting_plot(
    model_name: str = Query(..., description="Nom du modèle (ex: RandomForest, ExtraTrees...)"),
    metric: Optional[str] = Query(default=None, description="Nom de la métrique (ex: Accuracy ou R2). Si None -> auto.")
):
    _ensure_pipeline_loaded()

    if MUTING_CURVES_DSO2 is None or model_name not in MUTING_CURVES_DSO2:
        raise HTTPException(status_code=404, detail="Courbe introuvable. Vérifie model_name ou relance train avec progressive_muting.")

    df_curve = MUTING_CURVES_DSO2[model_name]

    if metric is None:
        metric = "Accuracy" if pipeline.problem_type == "classification" else "R2"

    if metric not in df_curve.columns:
        raise HTTPException(status_code=400, detail=f"Métrique '{metric}' non disponible. Colonnes: {df_curve.columns.tolist()}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_curve["Step"], df_curve[metric], marker="o", color="darkorange", linewidth=2, markersize=8)
    ax.set_title(f"Muting progressif — {model_name} — metric: {metric}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Step (nb features mutées)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _fig_to_png_response(fig)


# =========================
# PREDICTION (DSO1 ou DSO2)
# =========================
@app.post("/predict-row")
def predict_row(payload: PredictRow):
    _ensure_pipeline_loaded()
    try:
        df = pd.DataFrame([payload.features])

        if payload.use_optimized:
            if not hasattr(pipeline, "fitted_models_optimized_") or len(pipeline.fitted_models_optimized_) == 0:
                raise HTTPException(status_code=400, detail="Aucun modèle optimisé disponible. Exécutez d'abord le benchmark complet.")
            raise HTTPException(
                status_code=501,
                detail="Prédiction DSO2 non implémentée dans cette version. Utilisez DSO1."
            )
        else:
            pred = pipeline.predict(df, model_name=payload.model_name)
            return {
                "prediction": pred[0].item() if hasattr(pred[0], "item") else pred[0],
                "model_used": payload.model_name if payload.model_name else pipeline.best_model_name_,
                "approach": "DSO1"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(default=None),
    use_optimized: Optional[bool] = Query(default=False, description="Utiliser DSO2 si True, sinon DSO1")
):
    _ensure_pipeline_loaded()
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if pipeline.target_col in df.columns:
            df = df.drop(columns=[pipeline.target_col])

        if use_optimized:
            raise HTTPException(
                status_code=501,
                detail="Prédiction DSO2 non implémentée dans cette version. Utilisez DSO1."
            )
        else:
            preds = pipeline.predict(df, model_name=model_name)
            preds_list = [p.item() if hasattr(p, "item") else p for p in preds]
            return {
                "n_rows": len(preds_list),
                "model_used": model_name if model_name else pipeline.best_model_name_,
                "approach": "DSO1",
                "predictions": preds_list
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction fichier: {str(e)}")


# =========================
# EDA DASHBOARD
# =========================
@app.get("/eda/dashboard", response_class=HTMLResponse)
def eda_dashboard():
    df = _get_current_df()
    tcol = _get_target_col(df)
    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>EDA Dashboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 22px; }}
        h1 {{ margin: 0 0 6px; }}
        .muted {{ color: #555; margin-bottom: 16px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1200px; }}
        .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; }}
        img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #eee; }}
        a {{ color: #1565c0; }}
        code {{ background:#f6f6f6; padding:2px 6px; border-radius:6px; }}
      </style>
    </head>
    <body>
      <h1>EDA Dashboard</h1>
      <div class="muted">
        Dataset courant: <code>{CURRENT_FILENAME or "uploaded.csv"}</code> |
        Target: <code>{tcol}</code> |
        Problem: <code>{CURRENT_PROBLEM or "?"}</code> |
        <a href="/">Retour</a> | <a href="/benchmark-results">Résultats Benchmark</a>
      </div>

      <div class="grid">
        <div class="card">
          <h2>Matrice de corrélation (numérique)</h2>
          <img src="/eda/correlation.png" alt="Correlation matrix"/>
        </div>

        <div class="card">
          <h2>Boxplots (numérique)</h2>
          <img src="/eda/boxplots.png" alt="Boxplots"/>
        </div>

        <div class="card">
          <h2>Valeurs manquantes</h2>
          <img src="/eda/missing.png" alt="Missing values"/>
        </div>

        <div class="card">
          <h2>Distribution de la cible</h2>
          <img src="/eda/target.png" alt="Target distribution"/>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)


# =========================
# EDA PLOTS
# =========================
@app.get("/eda/correlation.png")
def eda_correlation():
    df = _get_current_df()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Pas assez de colonnes numériques pour calculer la corrélation.")

    corr = numeric_df.corr()
    n_cols = len(corr.columns)
    figsize = (max(5, n_cols * 0.3), max(4, n_cols * 0.28))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 6}
    )

    ax.set_title("Correlation Matrix", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    return _fig_to_png_response(fig)


@app.get("/eda/missing.png")
def eda_missing():
    df = _get_current_df()
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(11, 4))
    if len(miss) == 0:
        ax.text(0.5, 0.5, "Aucune valeur manquante", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.bar(miss.index.astype(str), miss.values)
        ax.set_title("Missing values per column")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    return _fig_to_png_response(fig)


@app.get("/eda/target.png")
def eda_target():
    df = _get_current_df()
    tcol = _get_target_col(df)
    y = df[tcol].dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    if y.nunique() <= 15:
        vc = y.value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"Target distribution (bar): {tcol}")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.hist(y, bins=30, edgecolor="black", alpha=0.85)
        ax.set_title(f"Target distribution (hist): {tcol}")
        ax.set_xlabel(tcol)
        ax.set_ylabel("Frequency")
    return _fig_to_png_response(fig)


@app.get("/eda/boxplots.png")
def eda_boxplots():
    df = _get_current_df()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Pas de colonnes numériques pour les boxplots.")

    n_cols = numeric_df.shape[1]
    n_rows = (n_cols // 3) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_df.columns):
        axes[i].boxplot(numeric_df[col].dropna())
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return _fig_to_png_response(fig)


@app.get("/eda/top.png")
def vi_top_png(n: int = 20):
    _ensure_pipeline_loaded()
    if getattr(pipeline, "vi_scores_", None) is None:
        raise HTTPException(status_code=400, detail="VI non disponible. Entraîne d'abord.")

    top = pipeline.vi_scores_.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top["Feature"], top["VI_Aggregate"], color="steelblue")
    ax.set_xlabel("Variable Importance")
    ax.set_title(f"Top {min(n, len(pipeline.vi_scores_))} Features (VI)")
    fig.tight_layout()
    return _fig_to_png_response(fig)


@app.get("/xai/status")
def xai_status():
    """Check XAI availability"""
    return {
        "xai_available": XAI_READY,
        "shap_installed": SHAP_AVAILABLE,
        "pipeline_loaded": pipeline is not None,
        "endpoints": {
            "heatmap": "/xai/rlt-heatmap.png",
            "shap": "/xai/shap-explanation.png?instance_idx=0&plot_type=bar",
            "shap_data": "/xai/shap-values?instance_idx=0",
            "comparison": "/xai/comparison.png",
            "dashboard": "/xai/dashboard"
        }
    }


@app.get("/xai/rlt-heatmap.png")
def xai_rlt_heatmap():
    """
    Generate RLT Feature Heatmap (Intrinsic Explanation)

    Returns PNG image showing feature importance as heatmap
    """
    if not XAI_READY or XAI_EXPLAINER is None:
        raise HTTPException(
            status_code=400,
            detail="XAI not ready. Train pipeline first with /upload-and-train"
        )

    try:
        buf = XAI_EXPLAINER.get_heatmap()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating heatmap: {str(e)}")


@app.get("/xai/shap-explanation.png")
def xai_shap_plot(
    instance_idx: int = Query(default=0, description="Test instance index to explain"),
    plot_type: str = Query(default="bar", description="Plot type: bar, waterfall, or force")
):
    """
    Generate SHAP explanation plot for a single instance

    Returns PNG image with SHAP values visualization
    """
    if not XAI_READY or XAI_EXPLAINER is None:
        raise HTTPException(
            status_code=400,
            detail="XAI not ready. Train pipeline first with /upload-and-train"
        )

    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="SHAP not installed. Install with: pip install shap"
        )

    try:
        buf, _ = XAI_EXPLAINER.get_shap_explanation(instance_idx, plot_type)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SHAP plot: {str(e)}")


@app.get("/xai/shap-values")
def xai_shap_values(instance_idx: int = Query(default=0)):
    """
    Get SHAP values as JSON for a single instance

    Returns:
        {
            "instance_idx": int,
            "shap_values": list[float],
            "base_value": float,
            "feature_values": list[float]
        }
    """
    if not XAI_READY or XAI_EXPLAINER is None:
        raise HTTPException(
            status_code=400,
            detail="XAI not ready. Train pipeline first with /upload-and-train"
        )

    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="SHAP not installed. Install with: pip install shap"
        )

    try:
        _, shap_data = XAI_EXPLAINER.get_shap_explanation(instance_idx, "bar")
        return {
            "status": "success",
            "instance_idx": instance_idx,
            **shap_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing SHAP values: {str(e)}")


@app.get("/xai/comparison.png")
def xai_comparison():
    """
    Compare RLT model vs Standard model (DSO1 best model)

    Returns PNG image with 3 comparison plots:
    - Predictions comparison
    - Feature importance comparison
    - Error distribution comparison
    """
    if not XAI_READY or XAI_EXPLAINER is None:
        raise HTTPException(
            status_code=400,
            detail="XAI not ready. Train pipeline first with /upload-and-train"
        )

    _ensure_pipeline_loaded()

    if pipeline.best_model_name_ is None or pipeline.best_model_name_ not in pipeline.fitted_models_:
        raise HTTPException(
            status_code=400,
            detail="No standard model available for comparison"
        )

    try:
        # Get standard model and data
        standard_model = pipeline.fitted_models_[pipeline.best_model_name_]

        # Need to get X_test and y_test - reconstruct from CURRENT_DF
        df = _get_current_df()
        target_col = _get_target_col(df)

        from sklearn.model_selection import train_test_split

        strat = df[target_col] if pipeline.problem_type == "classification" else None
        _, test_df = train_test_split(
            df,
            test_size=pipeline.test_size,
            random_state=pipeline.random_state,
            stratify=strat
        )

        X_test, y_test = pipeline.preprocess(test_df, target_col=target_col, fit=False)

        # Handle pipeline models (like RLT-ExtraTrees)
        if hasattr(standard_model, 'named_steps'):
            # It's a pipeline, extract the final model
            actual_model = standard_model.named_steps.get('model', standard_model)
        else:
            actual_model = standard_model

        # (CORRECTION) get_comparison() ne prend pas pipeline.problem_type en 4e argument
        buf = XAI_EXPLAINER.get_comparison(
            actual_model,
            X_test,
            y_test
        )

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating comparison: {str(e)}")


@app.get("/xai/dashboard", response_class=HTMLResponse)
def xai_dashboard():
    """
    XAI Dashboard - Interactive visualization of all XAI features
    """
    if not XAI_READY:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8"/>
            <title>XAI Dashboard - Not Ready</title>
            <style>
                body { font-family: Arial; padding: 40px; text-align: center; }
                .error { color: #d32f2f; font-size: 18px; }
                a { color: #1565c0; text-decoration: none; }
            </style>
        </head>
        <body>
            <h1>XAI Dashboard</h1>
            <p class="error">⚠️ XAI not initialized. Please train a model first.</p>
            <p><a href="/upload-and-train">Go to Training</a> | <a href="/">Home</a></p>
        </body>
        </html>
        """)

    shap_status = "✅ Available" if SHAP_AVAILABLE else "❌ Not installed (pip install shap)"

    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>XAI Dashboard - RLT Explainability</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 36px;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            .header p {{
                font-size: 16px;
                opacity: 0.9;
            }}
            .status {{
                background: #f5f5f5;
                padding: 20px 30px;
                border-bottom: 2px solid #e0e0e0;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .status-item {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .status-label {{
                font-weight: 600;
                color: #555;
            }}
            .status-value {{
                color: #667eea;
                font-weight: 500;
            }}
            .content {{
                padding: 30px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                color: #333;
                font-size: 24px;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }}
            .section p {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 20px;
            }}
            .viz-container {{
                background: #f9f9f9;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border: 2px solid #e0e0e0;
            }}
            .viz-container img {{
                width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .controls {{
                background: #fff;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
            }}
            .controls label {{
                font-weight: 600;
                color: #555;
                margin-right: 10px;
            }}
            .controls input, .controls select {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
            }}
            .controls button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                margin-left: 10px;
                transition: background 0.3s;
            }}
            .controls button:hover {{
                background: #5568d3;
            }}
            .nav {{
                padding: 20px 30px;
                background: #f5f5f5;
                border-top: 2px solid #e0e0e0;
                text-align: center;
            }}
            .nav a {{
                color: #667eea;
                text-decoration: none;
                margin: 0 15px;
                font-weight: 600;
                transition: color 0.3s;
            }}
            .nav a:hover {{
                color: #5568d3;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            @media (max-width: 768px) {{
                .grid-2 {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 XAI Dashboard</h1>
                <p>Explainable AI for RLT Models - Understand Your Predictions</p>
            </div>

            <div class="status">
                <div class="status-grid">
                    <div class="status-item">
                        <span class="status-label">XAI Status:</span>
                        <span class="status-value">✅ Ready</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">SHAP:</span>
                        <span class="status-value">{shap_status}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Problem Type:</span>
                        <span class="status-value">{CURRENT_PROBLEM or 'N/A'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Dataset:</span>
                        <span class="status-value">{CURRENT_FILENAME or 'N/A'}</span>
                    </div>
                </div>
            </div>

            <div class="content">
                <!-- RLT Heatmap -->
                <div class="section">
                    <h2>1️⃣ RLT Feature Heatmap (Intrinsic Explanation)</h2>
                    <p>
                        This heatmap shows the intrinsic feature importance computed by the RLT model.
                        Warmer colors (red) indicate higher importance, cooler colors (blue) indicate lower importance.
                    </p>
                    <div class="viz-container">
                        <img src="/xai/rlt-heatmap.png" alt="RLT Feature Heatmap" id="heatmap-img"/>
                    </div>
                </div>

                <!-- SHAP Explanation -->
                <div class="section">
                    <h2>2️⃣ SHAP Explanation (Post-hoc Analysis)</h2>
                    <p>
                        SHAP (SHapley Additive exPlanations) values show how each feature contributes to a specific prediction.
                        Select an instance and visualization type to explore.
                    </p>

                    <div class="controls">
                        <label for="instance-idx">Instance Index:</label>
                        <input type="number" id="instance-idx" value="0" min="0" step="1"/>

                        <label for="plot-type">Plot Type:</label>
                        <select id="plot-type">
                            <option value="bar">Bar Plot</option>
                            <option value="waterfall">Waterfall Plot</option>
                            <option value="force">Force Plot</option>
                        </select>

                        <button onclick="updateSHAP()">Update SHAP</button>
                    </div>

                    <div class="viz-container">
                        <img src="/xai/shap-explanation.png?instance_idx=0&plot_type=bar"
                             alt="SHAP Explanation" id="shap-img"/>
                    </div>
                </div>

                <!-- Model Comparison -->
                <div class="section">
                    <h2>3️⃣ RLT vs Standard Model Comparison</h2>
                    <p>
                        Compare the RLT model against the standard best model (DSO1).
                        This shows differences in predictions, feature importance, and error distribution.
                    </p>
                    <div class="viz-container">
                        <img src="/xai/comparison.png" alt="Model Comparison" id="comparison-img"/>
                    </div>
                </div>
            </div>

            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/eda/dashboard">📊 EDA Dashboard</a>
                <a href="/benchmark-results">📈 Benchmark Results</a>
                <a href="/status">ℹ️ Status</a>
            </div>
        </div>

        <script>
            function updateSHAP() {{
                const idx = document.getElementById('instance-idx').value;
                const plotType = document.getElementById('plot-type').value;
                const img = document.getElementById('shap-img');
                img.src = `/xai/shap-explanation.png?instance_idx=${{idx}}&plot_type=${{plotType}}&t=${{Date.now()}}`;
            }}

            // Auto-refresh images every 30 seconds
            setInterval(() => {{
                const heatmap = document.getElementById('heatmap-img');
                const comparison = document.getElementById('comparison-img');
                heatmap.src = heatmap.src.split('?')[0] + '?t=' + Date.now();
                comparison.src = comparison.src.split('?')[0] + '?t=' + Date.now();
            }}, 30000);
        </script>
    </body>
    </html>
    """

    return HTMLResponse(html)


# =========================
# INFO / STATUS
# =========================
@app.get("/")
def read_root():
    return {
        "message": "Backend API ready - DSO1 + DSO2 + XAI Support",
        "version": "5.0.0",
        "endpoints": {
            "train": "/upload-and-train",
            "benchmark": "/benchmark-results",
            "muting_results": "/dso2/muting-results",
            "muting_plot": "/dso2/muting-plot.png?model_name=RandomForest",
            "predict_row": "/predict-row",
            "predict_file": "/predict-file",
            "eda_dashboard": "/eda/dashboard",
            "xai_dashboard": "/xai/dashboard",
            "xai_status": "/xai/status",
            "xai_heatmap": "/xai/rlt-heatmap.png",
            "xai_shap": "/xai/shap-explanation.png",
            "xai_comparison": "/xai/comparison.png",
            "vi_plot": "/eda/top.png",
            "status": "/status"
        }
    }


@app.get("/status")
def get_status():
    return {
        "pipeline_loaded": pipeline is not None,
        "dataset_loaded": CURRENT_DF is not None,
        "current_filename": CURRENT_FILENAME,
        "current_target": CURRENT_TARGET,
        "current_problem": CURRENT_PROBLEM,
        "dso1": {
            "best_model": pipeline.best_model_name_ if pipeline else None,
            "models_available": list(pipeline.fitted_models_.keys()) if pipeline and hasattr(pipeline, "fitted_models_") else [],
            "benchmark_available": BENCHMARK_RESULTS_DSO1 is not None,
        },
        "dso2": {
            "best_model": pipeline.best_model_optimized_name_ if pipeline and hasattr(pipeline, "best_model_optimized_name_") else None,
            "models_available": list(pipeline.fitted_models_optimized_.keys()) if pipeline and hasattr(pipeline, "fitted_models_optimized_") else [],
            "benchmark_available": BENCHMARK_RESULTS_DSO2 is not None,
            "muting_available": MUTING_CURVES_DSO2 is not None and len(MUTING_CURVES_DSO2) > 0,
            "muting_models": list(MUTING_CURVES_DSO2.keys()) if MUTING_CURVES_DSO2 else [],
        }
    }
