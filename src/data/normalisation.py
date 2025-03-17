import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import MinMaxScaler
from check_structure import check_existing_file, check_existing_folder
import os

input_filepath = "./data/processed_data"
output_filepath = "./data/processed_data"

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath_X_train = f"{input_filepath}/X_train.csv"
    input_filepath_X_test = f"{input_filepath}/X_test.csv"

    process_data(input_filepath_X_train, input_filepath_X_test, output_filepath)
    
def process_data(input_filepath_X_train, input_filepath_X_test, output_filepath):
    # Import datasets
    df_X_train = import_dataset(input_filepath_X_train, sep=",")
    df_X_test = import_dataset(input_filepath_X_test, sep=",")

    # Normalisation
    X_train_scaled = normalisation(df_X_train)
    X_test_scaled = normalisation(df_X_test)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def normalisation(df):
    # Split data into training and testing sets
    scaler = MinMaxScaler()
    df_scaled_array = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled_array, columns=[df.columns])
    return df_scaled

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(input_filepath, output_filepath)