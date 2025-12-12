"""
PIPELINE TESTING SCRIPT
Test the production-ready RLT pipeline with CLI arguments
"""

import argparse
import pandas as pd
import numpy as np
import os
from model_RLT import RLTMLPipeline


workspace_path = r"/home/nour/Nour-Rajhi-4DS10-ml_Project"
dataset_path = os.path.join(workspace_path, "BostonHousing.csv")
model_path = "Boston_regression_model.pkl"


def run_preprocess():
    print("\n=== PREPROCESSING ===")

    df = pd.read_csv(dataset_path)

    pipeline = RLTMLPipeline(problem_type="regression", vi_threshold=0.01)

    X, y = pipeline.preprocess(df, target_col="medv", fit=True)
    print(f"✓ Preprocessing done: X={X.shape}, y={y.shape}")

    # Save intermediate objects
    pipeline.save_model(model_path)
    print(f"✓ Pipeline saved after preprocessing → {model_path}")


def run_training():
    print("\n=== TRAINING ===")

    df = pd.read_csv(dataset_path)

    pipeline = RLTMLPipeline(problem_type="regression", vi_threshold=0.01)

    # Preprocess before training
    X, y = pipeline.preprocess(df, target_col="medv", fit=True)
    model = pipeline.train(X, y, apply_muting=True)

    print("✓ Training complete")
    print(f"✓ Kept features: {len(pipeline.kept_features)}")

    # Save full model
    pipeline.save_model(model_path)
    print(f"✓ Model saved → {model_path}")


def run_prediction():
    print("\n=== PREDICTION ===")

    pipeline = RLTMLPipeline.load_model(model_path)
    print("✓ Model loaded")

    df = pd.read_csv(dataset_path)
    X, _ = pipeline.preprocess(df, target_col="medv", fit=False)

    preds = pipeline.predict(X.head(10))
    print(f"✓ Predictions: {preds[:5]}")


def run_load_test():
    print("\n=== MODEL LOAD CONSISTENCY TEST ===")

    df = pd.read_csv(dataset_path)

    pipeline = RLTMLPipeline(problem_type="regression", vi_threshold=0.01)

    # Preprocess + train
    X, y = pipeline.preprocess(df, target_col="medv", fit=True)
    pipeline.train(X, y, apply_muting=True)

    # Predictions from ORIGINAL pipeline
    preds_original = pipeline.predict(X.head(5))

    # Save model
    pipeline.save_model(model_path)

    # Reload model
    loaded = RLTMLPipeline.load_model(model_path)

    # Predictions from LOADED pipeline
    preds_loaded = loaded.predict(X.head(5))

    # Compare
    assert np.allclose(preds_original, preds_loaded), "❌ Predictions mismatch!"
    print("✓ Loaded model predictions match original")


def main():
    parser = argparse.ArgumentParser(description="RLT ML Pipeline Tester")

    parser.add_argument(
        "--preprocess", action="store_true", help="Exécuter preprocessing"
    )
    parser.add_argument("--train2", action="store_true", help="Entraîner la pipeline")
    parser.add_argument("--predict", action="store_true", help="Faire une prédiction")
    parser.add_argument(
        "--loadtest", action="store_true", help="Vérifier la cohérence modèle chargé"
    )

    args = parser.parse_args()

    if args.preprocess:
        run_preprocess()

    if args.train2:
        run_training()

    if args.predict:
        run_prediction()

    if args.loadtest:
        run_load_test()


if __name__ == "__main__":
    main()
