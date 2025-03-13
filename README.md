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
or by executing the preparation notebook, which downloads it automatically:
```
./notebooks/0 - preparations.ipynb
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
* ./dataset -> (gitignored) folder with the masked images dataset, either download it from here: https://drive.google.com/file/d/15T4543kcKJX6CzTcFfGGIqwHhSJmA_vM/view?usp=drive_link
or create the masked images yourself by executing '1 - masking.ipynb'
* ./notebooks -> the notebooks used to develop this model

### pipenv
We use pipenv for managing installed python libraries
* run 'pip install pipenv' to install pipenv, if not done yet
* create a virtual environment with 'pipenv install' (note: there is no recommended python version to ensure compatibility over many operating systems, everything above 3.9 should be fine)
* use "pipenv install \<library\>" instead of "pip install ..." in order to install python libraries, pipenv automatically updates the Pipfile dependencies
* run "pipenv install", whenever a change to Pipfile has been made in order to update your environment.
