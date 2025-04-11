# Import necessary Python modules
import os # to work with file and folder paths
import shutil # to move files and folders
import kagglehub # to download datasets from Kaggle Hub

COVID_FOLDER = "COVID-19_Radiography_Dataset" # Set the name of the folder where the dataset will be stored

def download_and_move_dataset(): # Set the destination path for the dataset (one level up from current folder)
    dest_path = os.path.join("..", COVID_FOLDER)

    # Create directories if they do not exist
    if os.path.exists(dest_path): # Check if the dataset folder already exists
        print("Dataset already exists.") # If yes, print message
        return # Stop the function here
    else:
        print("Dataset not found. Downloading...") # Otherwise, continue

    # Download the dataset from Kaggle Hub
    path = kagglehub.dataset_download(handle="tawsifurrahman/covid19-radiography-database", force_download=True) # name of the dataset

    # Move the dataset folder from the download location to the destination
    shutil.move(os.path.join(path, COVID_FOLDER), "..")
    print("Covid-19 Radiography Dataset downloaded and moved to parent directory.")
