"""
PIPELINE TESTING SCRIPT (CLI)
Compatible with model_RLT.py containing RLTBenchmarkPipeline


Usage examples:
  # Benchmark standard
  python main_rlt.py --benchmark --dataset BostonHousing.csv --target medv --problem regression --save Boston_pipeline.pkl


  # Benchmark optimisé (DSO2)
  python main_rlt.py --benchmark_opt --dataset BostonHousing.csv --target medv --problem regression --out_opt_results opt_results.csv


  # Benchmark optimisé + muting progressif + sauvegarde des courbes
  python main_rlt.py --benchmark_opt --dataset BostonHousing.csv --target medv --problem regression \
      --progressive_muting --out_muting_dir muting_curves/


  # EDA
  python main_rlt.py --eda --dataset BostonHousing.csv --target medv --problem regression


  # Predict (uniquement modèles benchmark() standard, pas DSO2)
  python main_rlt.py --predict --dataset BostonHousing.csv --target medv --problem regression --load Boston_pipeline.pkl --n 10
"""


import argparse
import os
import numpy as np
import pandas as pd

# =========================
# AJOUTS IMPORTS (XAI + split + modèles standard pour comparaison)
# =========================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# IMPORTANT: adapte l'import au nom exact de ton fichier
# - Si ton fichier s'appelle model_rlt.py -> from model_rlt import RLTBenchmarkPipeline
# - Si ton fichier s'appelle model_RLT.py -> from model_RLT import RLTBenchmarkPipeline
from model_RLT import (
    RLTBenchmarkPipeline,
    RLTXAIExplainer,
    
)



def read_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


# =========================
# AJOUT: utilitaire pour sauvegarder les PNG BytesIO renvoyés par l'XAI
# =========================
def write_bytesio_png(buf, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())


def run_eda(args):
    df = read_dataset(args.dataset)

    pipe = RLTBenchmarkPipeline(
        problem_type=args.problem,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        vi_threshold=args.vi_threshold,
    )

    pipe.data_understanding(df, target_col=args.target)


def run_benchmark(args):
    df = read_dataset(args.dataset)

    pipe = RLTBenchmarkPipeline(
        problem_type=args.problem,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        vi_threshold=args.vi_threshold,
    )

    results = pipe.benchmark(
        df,
        target_col=args.target,
        do_cv=(not args.no_cv),
        cv_folds=args.cv_folds,
        # =========================
        # AJOUT: possibilité de désactiver la VI depuis le CLI
        # =========================
        compute_vi=(not args.no_vi),
    )

    # Sauvegarde VI si disponible + si demandé
    if args.out_vi is not None:
        if hasattr(pipe, "vi_scores_") and pipe.vi_scores_ is not None:
            pipe.vi_scores_.to_csv(args.out_vi, index=False)
            print(f"✓ VI saved to: {args.out_vi}")
        else:
            print("⚠️  VI non disponible (vi_scores_ absent ou None).")

    print("\n=== BENCHMARK RESULTS ===")
    print(results.to_string(index=False))

    if args.out_results is not None:
        results.to_csv(args.out_results, index=False)
        print(f"\n✓ Results saved to: {args.out_results}")

    if args.save is not None:
        pipe.save(args.save)
        print(f"✓ Pipeline saved to: {args.save}")

    print(f"\n✓ Best model: {pipe.best_model_name_}")


def run_benchmark_optimized(args):
    """
    DSO2: Benchmark optimisé + option muting progressif.
    Note: Ces modèles optimisés ne sont pas encapsulés dans un pipeline sklearn unique,
    donc la prédiction "deploy" n'est pas gérée par run_predict() dans ce CLI.
    """
    df = read_dataset(args.dataset)

    pipe = RLTBenchmarkPipeline(
        problem_type=args.problem,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        vi_threshold=args.vi_threshold,
    )

    results_opt = pipe.benchmark_optimized(
        df,
        target_col=args.target,
        top_n=args.top_n,
        plot=args.opt_plot,
        progressive_muting=args.progressive_muting,
        muting_min_features=args.muting_min_features,
        muting_metric_plot=args.muting_metric_plot
    )

    print("\n=== OPTIMIZED BENCHMARK RESULTS (DSO2) ===")
    print(results_opt.to_string(index=False))
    if pipe.best_model_optimized_name_ is not None:
        print(f"\n✓ Best optimized model: {pipe.best_model_optimized_name_}")

    # Sauvegarde résultats DSO2
    if args.out_opt_results is not None:
        results_opt.to_csv(args.out_opt_results, index=False)
        print(f"\n✓ Optimized results saved to: {args.out_opt_results}")

    # Sauvegarde des courbes de muting progressif (1 CSV par modèle)
    if args.out_muting_dir is not None:
        os.makedirs(args.out_muting_dir, exist_ok=True)

        if hasattr(pipe, "muting_curves_optimized_") and pipe.muting_curves_optimized_:
            for model_name, df_curve in pipe.muting_curves_optimized_.items():
                out_path = os.path.join(args.out_muting_dir, f"muting_curve_{model_name}.csv")
                df_curve.to_csv(out_path, index=False)
                print(f"✓ Muting curve saved: {out_path}")
        else:
            print("⚠️  Aucune courbe de muting disponible (active --progressive_muting).")

    # Sauvegarde pipeline (pickle) si demandé
    if args.save is not None:
        pipe.save(args.save)
        print(f"✓ Pipeline saved to: {args.save}")


def run_predict(args):
    if args.load is None:
        raise ValueError("--load is required for --predict (pipeline must be loaded).")

    df = read_dataset(args.dataset)
    pipe = RLTBenchmarkPipeline.load(args.load)

    n = args.n if args.n is not None else 10
    sample = df.head(n).copy()

    preds = pipe.predict(sample, model_name=args.model_name)

    print("\n=== PREDICTION ===")
    print(f"Model used: {args.model_name if args.model_name else pipe.best_model_name_}")
    print("Predictions (first rows):")
    print(preds[: min(10, len(preds))])

    if args.proba:
        proba = pipe.predict_proba(sample, model_name=args.model_name)
        print("\nProbabilities (first rows):")
        print(proba[: min(5, len(proba))])


def run_load_test(args):
    """
    Train/benchmark -> save -> reload -> compare predictions consistency.
    """
    df = read_dataset(args.dataset)

    pipe = RLTBenchmarkPipeline(
        problem_type=args.problem,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        vi_threshold=args.vi_threshold,
    )

    _ = pipe.benchmark(df, target_col=args.target, do_cv=False)

    sample = df.head(20).copy()
    preds_original = pipe.predict(sample)

    save_path = args.save if args.save is not None else "pipeline.pkl"
    pipe.save(save_path)
    loaded = RLTBenchmarkPipeline.load(save_path)

    preds_loaded = loaded.predict(sample)

    if pipe.problem_type == "regression":
        ok = np.allclose(preds_original, preds_loaded)
    else:
        ok = np.array_equal(preds_original, preds_loaded)

    assert ok, "❌ Predictions mismatch after reload!"
    print("\n✓ Load test passed: predictions match after reload")


# =========================
# AJOUT: commande XAI (heatmap / SHAP / comparaison)
# =========================
def run_xai(args):
    """
    XAI (heatmap / SHAP / comparaison) basé sur RLTXAIExplainer (model_RLT.py).
    Génère des PNG (BytesIO) et les sauvegarde si un chemin est fourni.
    """
    df = read_dataset(args.dataset)

    # Split (stratify si classification)
    strat = df[args.target] if args.problem == "classification" else None
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strat,
    )

    pipe = RLTBenchmarkPipeline(
        problem_type=args.problem,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        vi_threshold=args.vi_threshold,
    )

    X_train, y_train = pipe.preprocess(train_df, target_col=args.target, fit=True)
    X_test, y_test = pipe.preprocess(test_df, target_col=args.target, fit=False)

    explainer = RLTXAIExplainer(
        problem_type=args.problem,
        vi_threshold=args.vi_threshold,
        random_state=args.seed,
    ).fit(X_train, y_train, X_test)

    did_any = False

    # Heatmap
    if args.xai_heatmap_out:
        buf = explainer.get_heatmap()
        write_bytesio_png(buf, args.xai_heatmap_out)
        print(f"✓ XAI heatmap saved: {args.xai_heatmap_out}")
        did_any = True

    # SHAP
    if args.xai_shap_out:
        buf, shap_info = explainer.get_shap_explanation(
            instance_idx=args.xai_instance,
            plot_type=args.xai_shap_plot_type,
        )
        write_bytesio_png(buf, args.xai_shap_out)
        print(f"✓ XAI SHAP plot saved: {args.xai_shap_out}")
        did_any = True

        if args.xai_shap_json_out:
            os.makedirs(os.path.dirname(args.xai_shap_json_out) or ".", exist_ok=True)
            pd.Series(shap_info).to_json(args.xai_shap_json_out)
            print(f"✓ XAI SHAP values saved: {args.xai_shap_json_out}")

    # Comparison (RLT vs Standard ExtraTrees sur l’espace standard préprocessé)
    if args.xai_compare_out:
        if args.problem == "classification":
            std = ExtraTreesClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
        else:
            std = ExtraTreesRegressor(n_estimators=200, random_state=args.seed, n_jobs=-1)

        std.fit(X_train, y_train)
        buf = explainer.get_comparison(std, X_test, y_test)
        write_bytesio_png(buf, args.xai_compare_out)
        print(f"✓ XAI comparison plot saved: {args.xai_compare_out}")
        did_any = True

    if not did_any:
        print("⚠️  --xai activé mais aucun output n’a été demandé. Utilise au moins un des flags:")
        print("   --xai_heatmap_out / --xai_shap_out / --xai_compare_out")


   


def build_parser():
    parser = argparse.ArgumentParser(
        description="RLT Benchmark Pipeline CLI (compatible with model_RLT.py)"
    )

    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--problem", type=str, choices=["classification", "regression"], required=True)

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vi_threshold", type=float, default=0.01)

    # Actions
    parser.add_argument("--eda", action="store_true", help="Run data understanding (EDA light)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark (train/test + optional CV)")
    parser.add_argument("--benchmark_opt", action="store_true", help="Run DSO2 optimized benchmark (feature engineering + embedded models)")
    parser.add_argument("--predict", action="store_true", help="Load a saved pipeline and predict on dataset head()")
    parser.add_argument("--loadtest", action="store_true", help="Train->save->load consistency test")
    parser.add_argument("--dso3", action="store_true", help="Run DSO3 simulation (4 scenarios)")

    # =========================
    # AJOUT: action XAI
    # =========================
    parser.add_argument("--xai", action="store_true", help="Run XAI (heatmap/SHAP/comparison)")

    # Benchmark options
    parser.add_argument("--no_cv", action="store_true", help="Disable CV during benchmark")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--out_results", type=str, default=None, help="CSV path to save benchmark results")
    parser.add_argument("--out_vi", type=str, default=None, help="CSV path to save variable-importance table")

    # =========================
    # AJOUT: option pour désactiver VI dans benchmark()
    # =========================
    parser.add_argument("--no_vi", action="store_true", help="Disable VI computation during benchmark()")

    # DSO2 options
    parser.add_argument("--top_n", type=int, default=15, help="DSO2: number of top features used to create linear combinations")
    parser.add_argument("--opt_plot", action="store_true", help="DSO2: show plots (barplot + optional muting curve)")
    parser.add_argument("--out_opt_results", type=str, default=None, help="CSV path to save DSO2 optimized results")


  
    # Progressive muting options
    parser.add_argument("--progressive_muting", action="store_true", help="DSO2: compute progressive muting curve per embedded model")
    parser.add_argument("--muting_min_features", type=int, default=5, help="DSO2 muting: keep at least this number of features")
    parser.add_argument("--muting_metric_plot", type=str, default=None,
                        help="DSO2 muting: metric name to plot (e.g. Accuracy or R2). Default auto.")
    parser.add_argument("--out_muting_dir", type=str, default=None,
                        help="Directory to save progressive muting curves (one CSV per model)")

    
    
    # Persistence / prediction options
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained pipeline (pickle)")
    parser.add_argument("--load", type=str, default=None, help="Path to load a trained pipeline (pickle)")
    parser.add_argument("--model_name", type=str, default=None, help="Force which fitted model to use for prediction")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to predict from dataset head(n)")
    parser.add_argument("--proba", action="store_true", help="Also print predict_proba (classification only)")

    # =========================
    # AJOUT: options XAI outputs
    # =========================
    parser.add_argument("--xai_heatmap_out", type=str, default=None, help="Output PNG path for RLT heatmap")
    parser.add_argument("--xai_shap_out", type=str, default=None, help="Output PNG path for SHAP explanation plot")
    parser.add_argument("--xai_shap_json_out", type=str, default=None, help="Output JSON path for SHAP values/base/feature_values")
    parser.add_argument("--xai_compare_out", type=str, default=None, help="Output PNG path for comparison plot (RLT vs Standard)")

    parser.add_argument("--xai_instance", type=int, default=0, help="Instance index for SHAP explanation")
    parser.add_argument(
        "--xai_shap_plot_type",
        type=str,
        default="bar",
        choices=["bar", "waterfall", "force"],
        help="SHAP plot type"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # =========================
    # AJOUT: inclure --xai dans la validation
    # =========================
    if not (args.eda or args.benchmark or args.benchmark_opt or args.predict or args.loadtest or args.xai):
        raise ValueError("Choose at least one action: --eda / --benchmark / --benchmark_opt / --predict / --loadtest / --xai")

    

    if args.eda:
        run_eda(args)

    if args.benchmark:
        run_benchmark(args)

    if args.benchmark_opt:
        run_benchmark_optimized(args)

    if args.predict:
        run_predict(args)

    if args.loadtest:
        run_load_test(args)

    # =========================
    # AJOUT: dispatcher XAI
    # =========================
    if args.xai:
        run_xai(args)
   


if __name__ == "__main__":
    main()
