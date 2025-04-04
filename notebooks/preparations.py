import os
import shutil
import kagglehub

COVID_FOLDER = "COVID-19_Radiography_Dataset"

def download_and_move_dataset():
    dest_path = os.path.join("..", COVID_FOLDER)

    # Create directories if they do not exist
    if os.path.exists(dest_path):
        print("Dataset already exists.")
        return
    else:
        print("Dataset not found. Downloading...")

    # Download dataset
    path = kagglehub.dataset_download(handle="tawsifurrahman/covid19-radiography-database", force_download=True)

    # Move dataset to parent directory
    shutil.copy(os.path.join(path, COVID_FOLDER), "..")
    print("Covid-19 Radiography Dataset downloaded and moved to parent directory.")
