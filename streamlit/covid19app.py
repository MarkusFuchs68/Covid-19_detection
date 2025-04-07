import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as ts

import os
import sys
# We need this in order to import the st_prediction module on streamlit community cloud
sys.path.append(os.path.join(os.path.dirname('st_prediction.py'), '..', 'streamlit'))
import st_prediction as pred
# Causes a reload of the module, so we can see the changes in the code without restarting the app
import importlib
importlib.reload(pred)


# Definitions
classes_4 = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
classes_2 = ['Normal', 'COVID']
MODEL_FOLDER = os.path.join('.', 'models')


def download_models():
    import gdown
    from time import sleep

    # Destination folder where files will be downloaded
    os.makedirs(MODEL_FOLDER, exist_ok=True)  # create if it doesn't exist

    # List all files first to know how many we will download
    drive_url = 'https://drive.google.com/drive/folders/1-CXGnmQyunu2qF1fxT1Ja_F_qeM65XEk'
    file_list = gdown.download_folder(drive_url, output=MODEL_FOLDER,
                                      quiet=True, use_cookies=False,
                                      remaining_ok=True, skip_download=True)

    if not file_list:
        st.error("No files found or invalid folder URL.")
        return

    st.write(f"Found {len(file_list)} files to download.")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Download one by one
    for i, file in enumerate(file_list):

        # Update status text
        status_text.text(f"Downloading: {os.path.basename(file.path)} ({i+1}/{len(file_list)})")
        sleep(0.1)  # just to see the progress clearly

        url = f"https://drive.google.com/uc?id={file.id}"
        gdown.download(url=url, output=file.local_path, quiet=True, use_cookies=False)

        # Update progress bar
        percent_complete = (i + 1) / len(file_list)
        progress_bar.progress(percent_complete)


# The streamlit app code:
st.title('Covid-19 Data Analysis')
st.write('This is a simple Streamlit app to analyze Covid-19 chest X-ray images by various predefined models.')

st.sidebar.title('Navigation')
pages = ['Retrieve models', 'Model selection', 'Prediction']
page = st.sidebar.radio('Select a page:', pages)

if page == 'Retrieve models':
    st.subheader('Retrieve models')

    st.write('Current models available:')
    if os.path.isdir('./models'):
        model_list = os.listdir('./models')
        model_names = [model_filename.split('.')[0] for model_filename in model_list if model_filename.endswith('.keras')]
        model_names = sorted(model_names)
        for model_filename in model_names:
            st.write(model_filename)

    st.write('Click the button below to download and update the available models.')
    if st.button('Download models'):
        # We download the models from our Google Drive
        st.write('Downloading models from Google Drive...')
        st.write('Please wait...')
        download_models()
        st.write('Models downloaded successfully. Please refresh the page to see the models.')

elif page == 'Model selection':
    st.subheader('Model Selection')
    st.write('Select a model to use for prediction:')
    model_list = os.listdir('./models')
    model_names = [model_filename.split('.')[0] for model_filename in model_list if model_filename.endswith('.keras')]
    model_names = sorted(model_names)
    if st.session_state.get('model_name'):
        selected_model = st.session_state['model_name']
    else:
        selected_model = model_names[0] if model_names else None
    selected_model = st.selectbox('Select a model:', model_names, index=model_names.index(selected_model) if selected_model in model_names else 0)
    if selected_model is None:
        st.error('No models available. Please download models first.')
        st.stop()

    st.write(f'You selected: {selected_model}')

    # Load the selected model
    model = ts.keras.models.load_model(os.path.join(MODEL_FOLDER, selected_model + '.keras'))
    st.write('Model loaded successfully.')
    st.write('Model summary:')
    df = pred.model_summary_to_df(model)
    st.write(df)

    # Save it in streamlit session state
    classes = classes_2 if '2-classes' in selected_model else classes_4
    st.session_state['model_name'] = selected_model
    st.session_state['model'] = model
    st.session_state['classes'] = classes


elif page == 'Prediction':
    st.subheader('Prediction')

    # First check if we have models loaded
    if 'model' not in st.session_state:
        st.error('No model loaded. Please select a model first.')
        st.stop()

    model_name = st.session_state['model_name']
    model = st.session_state['model']
    classes = st.session_state['classes']

    st.write('Enter a URL of an X-ray image for prediction:')
    image_url = st.text_input('Image URL:', value='https://content.ca.healthwise.net/resources/14.1/en-ca/media/medical/hw/h9991297_001.jpg')
    if not image_url:
        st.stop()

    st.write(f'You entered: {image_url}')
    image = pred.load_image_from_url(image_url)

    # Show the original image
    st.image(image)

    # Prepare the image for prediction in the selected model
    img_prepared = pred.prepare_image_for_model(image, model_name, model)

    # Predict using the model
    st.write('Predicting...')
    pred_df = pred.predict_image(img_prepared, model_name, model, classes)
    st.dataframe(pred_df)

    st.write('Check the following checkbox to show a Grad-CAM of the prediction:')
    show_gradcam = st.checkbox('Show Grad-CAM')
    if show_gradcam:
        pred.show_feature_maps(img_prepared, model_name, model)
