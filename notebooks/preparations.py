import os
import shutil
import kagglehub

def download_and_move_dataset():
    # Create directories if they do not exist
    if os.path.exists("COVID-19_Radiography_Dataset"):
        print("Dataset already exists.")
        return True
    else:
        print("Dataset not found.")
    os.makedirs("COVID-19_Radiography_Dataset", exist_ok=True)
    # Download dataset
    path = kagglehub.dataset_download(handle="tawsifurrahman/covid19-radiography-database", force_download=True)
    # Move dataset to parent directory
    shutil.move(os.path.join(path, 'COVID-19_Radiography_Dataset'), "..")
    print("Covid-19 Radiography Dataset downloaded and moved to parent directory.")
