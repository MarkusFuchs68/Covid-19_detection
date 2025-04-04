import os
import shutil
import kagglehub

def check_for_dataset(foldername):
    # Check if the dataset directory exists
    if os.path.exists("COVID-19_Radiography_Dataset"):
        print("Dataset already exists.")
        return True
    else:
        print("Dataset not found.")
        return False

def download_and_move_dataset():
    # Create directories if they do not exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    # Download and move the dataset
    download_and_move_covid19_radiography_dataset()

# Download dataset
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
# Move dataset to parent directory
shutil.move(os.path.join(path, 'COVID-19_Radiography_Dataset'), "..")
print("Covid-19 Radiography Dataset downloaded and moved to parent directory.")

