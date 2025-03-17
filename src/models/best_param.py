import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd 
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

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

    rf_regressor = ensemble.RandomForestRegressor(random_state=42)

    # Définir la grille d'hyperparamètres
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Lancer la recherche des meilleurs hyperparamètres
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Récupérer les meilleurs paramètres
    best_params = grid_search.best_params_
    print("Meilleurs paramètres :", best_params)

    # Sauvegarder les meilleurs paramètres dans un fichier .pkl
    with open(f"{output_filepath}/best_rf_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    print("Parameters saved successfully.")

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(input_filepath, output_filepath)