# Covid-19_detection
In this repository we collaborate on the Covid-19 detection project, which develops a deep learning convolutional neural network model with the intention to detect a Covid-19 disease from a given chest x-ray-image.

## Usage

### Download Initial Data

This project uses the original dataset from kaggle, you need to download it from here: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data. 

Either manually by yourself:
```
curl -L -o covid19-radiography-database.zip \
    https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-databas
unzip covid19-radiography-database.zip
```
or by executing
```
import preparations as prep
prep.download_and_move_dataset()
```
furtheron all notebooks/code is expecting the original dataset in (.gitignored) folder:

./COVID-19_Radiography_Dataset


### Docker
```
docker compose up
```

or 

```
docker compose -f docker-compose-gpu.yaml up
```

Open [backend](http://localhost:8888) in browser. Make sure, that nothing blocks port 8888 (!).

### Folder structure
* ./COVID-19_Radiography_Dataset -> the downloaded (gitignored) original x-ray-image dataset
* ./dataset -> (gitignored) folder with the generated masked images dataset, either download it from [here](https://drive.google.com/file/d/15T4543kcKJX6CzTcFfGGIqwHhSJmA_vM/view?usp=drive_link)
or create the masked images yourself by executing '1 - masking.ipynb'
* ./models -> (gitignored) folder with the trained models, download it from [here](https://drive.google.com/drive/folders/1_i8YZdClF5pnDEeyZL17IR5EAcqhEzUR)
or create the masked images yourself by executing '1 - masking.ipynb'
* [./notebooks](./notebooks) -> the notebooks used to develop this model, incl. reusable python scripts
* [./streamlit](./streamlit) -> the accompanying streamlit application

### pipenv
We use pipenv for managing installed python libraries
* run 'pip install pipenv' to install pipenv, if not done yet
* create a virtual environment with 'pipenv install' (note: there is no recommended python version to ensure compatibility over many operating systems, everything above 3.9 should be fine)
* use "pipenv install \<library\>" instead of "pip install ..." in order to install python libraries, pipenv automatically updates the Pipfile dependencies
* run "pipenv install", whenever a change to Pipfile has been made in order to update your environment.

### [Notebooks](./notebooks)
* [0 - inspection](./notebooks/0%20-%20inspection.ipynb): first inspection of the dataset and display of example images of all classes
* [1 - masking](./notebooks/1%20-%20masking.ipynb): generation of masked x-ray images by overlaying the masks on the x-ray images, saved in ./dataset folder
* [2 - modelling](./notebooks/2%20-%20modelling.ipynb): first quick model on a reduced dataset for experimenting
* [2*](./notebooks): further notebooks with different models, each including the learning curve, the model performance and confustion matrix report
* [3 - experiments](./notebooks/3%20-%20experiments.ipynb): having a deeper look into the models from 2*-notebooks in order to visually understand, what happens on the convolutional layers
* [3*](./notebooks): more models with adapted strategies (= learnings from 2*-notebooks)
* [4*](./notebooks): finetuned models with adapted strategies (= learnings from 3*-notebooks)
* [5 - predictions](./notebooks/5%20-%20prediction.ipynb): predictions and comparism made by the models on random images taken from the internet

### Notes
Due to its size on disk the models generated were saved on a private Google space, hence the notebook 5 with the predictions cannot be run from the repo directly, you must first download the trained models from [here](https://drive.google.com/drive/folders/1_i8YZdClF5pnDEeyZL17IR5EAcqhEzUR) into the [models](./models) folder.

### Streamlit App
You can run the accompanying streamlit app locally with:
```
streamlit run ./streamlit/covid19app.py
```