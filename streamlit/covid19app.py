import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as ts

import os
import st_content as content
import st_prediction as pred
import importlib
importlib.reload(content)
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
st.title('Covid-19 chest X-ray image Analysis')

st.sidebar.title('Navigation')
pages = ['Home', 'Analysis', 'Modelisation', 'Model selection', 'Prediction', 'About']
page = st.sidebar.radio('Go to:', pages)

if page == 'Home':

    st.markdown(content.home_general)
    st.markdown(content.home_context)
    st.markdown(content.home_samples)
    st.markdown(content.home_normal)
    st.image(os.path.join('content', 'normal.png'), caption='Healthy lungs')
    st.markdown(content.home_viral_pneumonia)
    st.image(os.path.join('content', 'viral_pneumonia.png'), caption='Viral pneumonia')
    st.markdown(content.home_lung_opacity)
    st.image(os.path.join('content', 'lung_opacity.png'), caption='Lung opacity')
    st.markdown(content.home_covid)
    st.image(os.path.join('content', 'covid.png'), caption='Covid-19')

elif page == 'Analysis':

    st.markdown(content.analysis_structure_1)
    st.image(os.path.join('content', 'class_distribution.png'), caption='Class distribution')
    st.markdown(content.analysis_structure_2)
    st.markdown(content.analysis_masking)
    st.image(os.path.join('content', 'masking.png'), caption='Image masking')
    st.markdown(content.analysis_structure_3)
    st.markdown(content.analysis_challenges)
    st.markdown(content.analysis_biases)

elif page == 'Modelisation':

    st.markdown(content.modelisation_intro)

    df_models = pd.DataFrame(content.modelisation_models['models'])
    st.dataframe(df_models)

elif page == 'Model selection':

    # Always offer the download or update of the models
    if st.button('Download/update models'):
        # We download the models from our Google Drive
        st.write('Downloading models from Google Drive...')
        st.write('Please wait...')
        download_models()
        st.write('Models downloaded successfully. Refreshing page...')
        st.rerun()

    # Check if the models folder exists
    if not os.path.isdir(MODEL_FOLDER):
        st.error('No models available. Please download models first.')
        st.stop()

    # Load the models folder
    model_list = os.listdir(MODEL_FOLDER)
    model_names = [model_filename.split('.')[0] for model_filename in model_list if model_filename.endswith('.keras')]
    model_names = sorted(model_names)

    # Check if we have models in the folder
    if len(model_names) == 0:
        st.error('No models available. Please download models first.')
        st.stop()

    # Let the user select one or preset it with the last selected one
    selected_model = st.session_state.get('model_name') # Returns None, if never selected
    selected_model = st.selectbox('Select a model for prediction:', model_names, index=model_names.index(selected_model) if selected_model in model_names else None)
    if selected_model is None:
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

    # First check if we have models loaded
    if 'model' not in st.session_state:
        st.error('No model loaded. Please select a model first.')
        st.stop()

    model_name = st.session_state['model_name']
    model = st.session_state['model']
    classes = st.session_state['classes']

    # Let the user choose between file upload or URL input
    image = None
    loading_type = 0
    st.markdown(content.prediction_note)
    url_input, divider, file_input = st.columns([3, 1, 3])
    
    with divider:
        st.markdown(content.prediction_or, unsafe_allow_html=True)

    with url_input:
        st.write('Enter a URL of an X-ray image for prediction:')
        image_url = st.text_input('Image URL:', value='https://content.ca.healthwise.net/resources/14.1/en-ca/media/medical/hw/h9991297_001.jpg')
        if image_url and image_url != '':
            image = pred.load_image_from_url(image_url)
            st.write(f'You entered: {image_url}')
            loading_type = 1

    with file_input:
        st.write('Upload an X-ray image for prediction:')
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = pred.load_image_from_file(uploaded_file)
            st.write(f'You uploaded: {uploaded_file.name}')
            loading_type = 2

    # Show the original image
    if image is None:
        st.stop()

    # Predict using the model
    if loading_type == 1:
        st.write('Predicting the following image from URL...')
    elif loading_type == 2:
        st.write('Predicting the following image from file...')

    # Show the loaded image
    st.image(image, width=300)

    # Prepare the image for prediction in the selected model
    img_prepared = pred.prepare_image_for_model(image, model_name, model)

    # Predict the prepared image
    pred_df = pred.predict_image(img_prepared, model_name, model, classes)
    st.dataframe(pred_df)

    # Optionally show a Grad-CAM
    st.write('Check the following checkbox to show a Grad-CAM of the prediction:')
    show_gradcam = st.checkbox('Show Grad-CAM')
    if show_gradcam:
        pred.show_feature_maps(img_prepared, model_name, model)

elif page == 'About':

    st.markdown(content.about)
