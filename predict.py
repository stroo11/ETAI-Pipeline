"""
Predict a CSV with the model saved by main.py -- week 10. No training: the saved pipeline holds every value learned in
training (medians, encodings, the selected features, the trees), and new rows go through exactly the same steps.

Run from the project folder:
    python main.py                                                    # train, evaluate, save models/model.joblib
    python predict.py data/compas_new_people.csv predictions.csv      # predict

    python predict.py data/adult_new.csv adult_predictions.csv --config config_adult.yaml

One output row per input row (duplicates and rows with gaps included): id, probability, prediction.
Only load model files you trust: a .joblib file can run code when it is loaded.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.predict import library_versions, predict


def check_versions(saved: dict) -> None:
    """Warns when a library version differs from the one the model was saved with (it may not behave the same)."""
    installed = library_versions()
    for name, version in (saved or {}).items():
        if installed.get(name) != version:
            print(f"Warning: the model was saved with {name} {version}, but {installed.get(name)} is installed.")


def main():
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    saved = joblib.load(config["output"]["model_path"])
    check_versions(saved["versions"])

    rows = pd.read_csv(args.input)
    predictions = predict(rows, saved["pipeline"], config, columns=saved["columns"])
    predictions.to_csv(args.output, index=False)
    print(f"{len(rows)} rows in, {len(predictions)} rows predicted -> {args.output}   "
          f"(model saved {saved['saved']}, {int(predictions['prediction'].sum())} predicted positive)")


if __name__ == "__main__":
    main()
