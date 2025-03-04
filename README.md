# Covid-19_detection
In this repository we collaborate on the Covid-19 detection project, which develops a deep learning convolutional neural network model with the intention to detect a Covid-19 disease from a given chest x-ray-image.

### This repo holds the following folders:
* ./venv -> (gitignored) local Python environment, the installed libraries are found in ./requirements.txt; use "pip install -r requirements.txt" to create the same environment, here based on Python 3.9.6* ./notebooks -> the jupyter notebooks
* ./COVID-19_Radiography_Dataset -> (gitignored) original dataset from kaggle, you need to download it from here: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data 
* ./dataset -> (gitignored) folder with the masked images dataset, either download it from here: https://drive.google.com/file/d/15T4543kcKJX6CzTcFfGGIqwHhSJmA_vM/view?usp=drive_link
or create the masked images yourself by executing '1 - masking.ipynb'
* ./notebooks -> the notebooks used to develop this model
