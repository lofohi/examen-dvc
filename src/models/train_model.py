import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd 
from sklearn import ensemble
import joblib

input_filepath = "./data/processed_data"
output_filepath = "./models"

def main(input_filepath, output_filepath):
    input_filepath_X_train_scaled = f"{input_filepath}/X_train_scaled.csv"
    input_filepath_y_train = f"{input_filepath}/y_train.csv"

    def import_dataset(file_path, **kwargs):
        return pd.read_csv(file_path, **kwargs)
    
    X_train_scaled = import_dataset(input_filepath_X_train_scaled, sep=",")
    y_train = import_dataset(input_filepath_y_train)
    y_train = np.ravel(y_train)


    # Charger les meilleurs paramtres
    with open(f"{output_filepath}/best_rf_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    # Initialiser un modèle RandomForestRegressor avec les meilleurs paramètres
    rf_regressor = ensemble.RandomForestRegressor(**best_params, random_state=42)

    # Entraîner le modèle
    rf_regressor.fit(X_train_scaled, y_train)

    # Sauvegarder le modèle
    joblib.dump(rf_regressor, f"{output_filepath}/trained_model.joblib")
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(input_filepath, output_filepath)