import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd 
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

input_filepath = "./data/processed_data"
output_filepath = "./models"

def main(input_filepath, output_filepath):
    input_filepath_X_test_scaled = f"{input_filepath}/X_test_scaled.csv"
    input_filepath_y_test = f"{input_filepath}/y_test.csv"

    def import_dataset(file_path, **kwargs):
        return pd.read_csv(file_path, **kwargs)
    
    X_test_scaled = import_dataset(input_filepath_X_test_scaled, sep=",")
    y_test = import_dataset(input_filepath_y_test)
    y_test = np.ravel(y_test)

    # Charger le modèle
    model = joblib.load(f"{output_filepath}/trained_model.joblib")

    # Prédictions
    predictions = model.predict(X_test_scaled)

    # Sauvegarder les prédictions dans /data
    df_predictions = pd.DataFrame({
        "Actual": y_test,           # Valeurs réelles
        "Predicted": predictions    # Valeurs prédites par le modèle
    })
    csv_path = "data/predictions.csv"
    df_predictions.to_csv(csv_path, index=False)
    print("Les prédictions ont bien été enregistrées.")
    
    # Evaluation du modèle
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    scores = {"MSE": mse, "R2 Score": r2}
    score_path = Path("./metrics/scores.json")
    score_path.write_text(json.dumps(scores))
    print("Les scores ont bien été enregistrés.")
    
if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main(input_filepath, output_filepath)