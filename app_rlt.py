from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from model_RLT import RLTMLPipeline
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


MODEL_PATH = "Boston_regression_model.pkl"
DATA_PATH = "BostonHousing.csv"

# -------------------------------
# Chargement pipeline RLT
# -------------------------------
if os.path.exists(MODEL_PATH):
    pipeline = RLTMLPipeline.load_model(MODEL_PATH)
else:
    pipeline = None

# -------------------------------
# Création FastAPI
# -------------------------------
app = FastAPI(
    title="RLT ML Pipeline API",
    description="API pour prédire et réentraîner le modèle RLT",
    version="1.0.0",
)

# Monter le dossier 'static' pour servir les fichiers HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# Schémas JSON
# -------------------------------
class InputData(BaseModel):
    # On laisse toutes les colonnes possibles de ton dataset Boston
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float


class RetrainParams(BaseModel):
    target_col: str = "medv"
    vi_threshold: float = 0.01
    apply_muting: bool = True


# -------------------------------
# Endpoint /predict
# -------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        if pipeline is None:
            raise HTTPException(status_code=400, detail="Modèle non chargé")

        df = pd.DataFrame([data.dict()])
        X_scaled, _ = pipeline.preprocess(df, target_col=None, fit=False)
        pred = pipeline.predict(X_scaled)

        return {"medv": float(pred[0])}

    except Exception as e:
        # pour debug temporaire, tu peux aussi imprimer la stack trace ici
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}"
        )

# -------------------------------
# Endpoint /retrain
# -------------------------------
@app.post("/retrain")
def retrain(params: RetrainParams):
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail="Fichier de données introuvable")

        df = pd.read_csv(DATA_PATH)

        # Supprimer lignes avec NaN dans y
        df = df.dropna(subset=[params.target_col])

        # Initialiser pipeline
        global pipeline
        pipeline = RLTMLPipeline(problem_type="regression", vi_threshold=params.vi_threshold)

        # Preprocessing
        X_scaled, y = pipeline.preprocess(df, target_col=params.target_col, fit=True)

        # Entraînement
        pipeline.train(X_scaled, y, apply_muting=params.apply_muting)

        # Sauvegarde
        pipeline.save_model(MODEL_PATH)

        return {"status": "success", "message": "Modèle RLT réentraîné et sauvegardé."}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant le réentraînement : {str(e)}"
        )
    
@app.post("/upload-and-train")
async def upload_and_train(
    file: UploadFile = File(...),
    vi_threshold: float = 0.01,
    apply_muting: bool = True,
    target_col: str = "medv"
):
    try:
        # Lire le CSV uploadé directement dans pandas
        df = pd.read_csv(file.file)

        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonne cible '{target_col}' absente du CSV")

        # Supprimer lignes avec NaN dans la cible
        df = df.dropna(subset=[target_col])

        global pipeline
        pipeline = RLTMLPipeline(problem_type="regression", vi_threshold=vi_threshold)

        # Préprocessing + entraînement
        X_scaled, y = pipeline.preprocess(df, target_col=target_col, fit=True)
        pipeline.train(X_scaled, y, apply_muting=apply_muting)

        # Calcul des métriques de régression sur le même dataset
        X_metrics = X_scaled[pipeline.kept_features]
        y_pred = pipeline.model.predict(X_metrics)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)

        # Sauvegarde du modèle
        pipeline.save_model(MODEL_PATH)

        return {
            "status": "success",
            "message": "Modèle entraîné à partir du fichier uploadé.",
            "metrics": {
                "r2": float(r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement : {str(e)}")


@app.get("/")
def read_index():
    return RedirectResponse(url="/static/index.html")

