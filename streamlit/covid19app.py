# Import Streamlit and other required libraries
import select
import streamlit as st # Web app UI with Streamlit
import numpy as np          # Numerical operations
import pandas as pd # For data handling
import st_content as content # Custom module for UI content
import st_prediction as pred # Custom module for prediction
import tensorflow as ts # For deep learning
import os # For file/folder handling
import sys # For system-specific operations
# Add the "streamlit" folder to the system path
# This allows us to import files from this directory
streamlit_dir = os.path.dirname(__file__)
sys.path.append(streamlit_dir)

# Define the class names (4-class and 2-class variants)
classes_4 = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
classes_2 = ['Normal', 'COVID']
MODEL_FOLDER = os.path.join('.', 'models') # Set the folder path where models will be saved


def download_models(): # Function to download model files
    import gdown # gdown helps download from Google Drive
    from time import sleep # sleep allows us to pause the loop if needed

    # Destination folder where files will be downloaded
    os.makedirs(MODEL_FOLDER, exist_ok=True)  # create if it doesn't exist

    # List all files first to know how many we will download
    drive_url = 'https://drive.google.com/drive/folders/1-CXGnmQyunu2qF1fxT1Ja_F_qeM65XEk' # Define the URL to the Google Drive folder where models are stored
    file_list = gdown.download_folder(drive_url, output=MODEL_FOLDER, # Use gdown to list the files in the folder # Google Drive folder URL # Where to store the files
                                      quiet=True, use_cookies=False, # Don’t print download info # Don’t use cookies
                                      remaining_ok=True, skip_download=True) # Continue if some files remain # Just list files, don’t download yet

    if not file_list: # If there are no files, print error and stop
        st.error("No files found or invalid folder URL.")
        return

    st.write(f"Found {len(file_list)} files to download.") # If files are found, show how many
    progress_bar = st.progress(0) # Initialize progress bar
    status_text = st.empty() # Placeholder for status updates

    # Download one by one
    for i, file in enumerate(file_list): # Loop through and download each file one by one

        # Update status text
        status_text.text(f"Downloading: {os.path.basename(file.path)} ({i+1}/{len(file_list)})")
        sleep(0.1)  # just to see the progress clearly

        url = f"https://drive.google.com/uc?id={file.id}" # Create the URL to download the file directly from Google Drive
        gdown.download(url=url, output=file.local_path, quiet=True, use_cookies=False) # Start downloading the file

        # Update progress bar
        percent_complete = (i + 1) / len(file_list) # Update the progress bar (percent complete)
        progress_bar.progress(percent_complete)


# The streamlit app code:
st.title('Covid-19 chest X-ray image Analysis') # Title of the web app

st.sidebar.title('Navigation') # Sidebar for navigation
pages = ['Home', 'Analysis', 'Modelisation', 'Model selection', 'Prediction', 'About'] # List of pages user can choose from in the app
page = st.sidebar.radio('Go to:', pages) # Create a radio button to navigate between pages

if page == 'Home': # If user selected the "Home" page
    # Show introductory text and explanations from content module
    st.markdown(content.home_general)
    st.markdown(content.home_context)
    st.markdown(content.home_samples)
    st.markdown(content.home_normal) # Section for showing healthy (normal) lung images
    st.image(os.path.join(streamlit_dir, 'content', 'normal.png'), caption='Healthy lungs')
    st.markdown(content.home_viral_pneumonia) # Section for viral pneumonia examples
    st.image(os.path.join(streamlit_dir, 'content', 'viral_pneumonia.png'), caption='Viral pneumonia')
    st.markdown(content.home_lung_opacity) # Section for lung opacity examples
    st.image(os.path.join(streamlit_dir, 'content', 'lung_opacity.png'), caption='Lung opacity')
    st.markdown(content.home_covid) # Section for COVID-19 examples
    st.image(os.path.join(streamlit_dir, 'content', 'covid.png'), caption='Covid-19')

elif page == 'Analysis': # If user selected the "Analysis" page
    # Show structure of the analysis steps
    st.markdown(content.analysis_structure_1)
    st.image(os.path.join(streamlit_dir, 'content', 'class_distribution.png'), caption='Class distribution')
    st.markdown(content.analysis_structure_2) # Continue with more analysis explanations
    st.markdown(content.analysis_masking) # Show image masking explanation
    st.image(os.path.join(streamlit_dir, 'content', 'masking.png'), caption='Image masking')
    st.markdown(content.analysis_structure_3) # More sections of the analysis
    st.markdown(content.analysis_challenges)
    st.markdown(content.analysis_biases)

elif page == 'Modelisation': # If the selected page is 'Modelisation'

    st.markdown(content.modelisation_intro) # Show introductory text about modelisation

    with st.expander('Variable descriptions'): # Expandable section to show variable explanations
        st.markdown(content.modelisation_variables)

    # Read in our model summary
    df_models = pd.DataFrame(content.modelisation_model_summary)  # Load a table with all model summaries

    # and rearrange the column order to our wishes
    df_models_summary = df_models[content.modelisation_summary_columns] # Rearrange the order of the columns to fit our layout preference
    st.dataframe(df_models_summary, use_container_width=False)

    # Individual model summaries
    st.markdown(content.modelisation_details)  # Show detailed information below

    # Data for the following model detail description
    df_model_details = pd.DataFrame(content.modelisation_model_details).set_index('name') # Load the full model detail information into a DataFrame

    # This lists all our model names
    model_list_detail = df_models['name'].tolist() # Get a list of model names
    model_list_detail = sorted(model_list_detail) # Sort alphabetically
    selected_model_detail = st.selectbox("Select a model:", model_list_detail, key='model_detail')  # Dropdown menu to select a model
    if selected_model_detail is not None: # If a model is selected, show its information
        # We show the model description, the loss and accuracy curves, 
        # the performance report on the validation dataset, and the confusion matrix
        st.markdown('**Model report for model:** ' + selected_model_detail) # Title for selected model

        # The description and underlying ideas
        st.markdown(df_model_details.loc[selected_model_detail]['description']) # Show model description

        # The conclusion
        st.markdown(df_model_details.loc[selected_model_detail]['conclusion']) # Show model conclusion

        # The model learning curve
        st.markdown('**Training History:**') # Show training curve image
        performance_path = os.path.join(streamlit_dir, 'content', selected_model_detail + '_training.png')
        st.image(performance_path)

        # The model performance
        st.markdown('**Performance Report / Recall normalized Confusion Matrix:**') # Show confusion matrix and performance summary
        performance_path = os.path.join(streamlit_dir, 'content', selected_model_detail + '_performance.png')
        st.image(performance_path)

    st.markdown(content.modelisation_learnings)

elif page == 'Model selection': # If the selected page is 'Model selection'

    # Always offer the download or update of the models
    if st.button('Download/update models'): # Button to allow the user to download or update the model files
        # We download the models from our Google Drive
        st.write('Downloading models from Google Drive...') # Display a message to the user
        st.write('Please wait...')
        download_models() # Call the function to download models
        st.write('Models downloaded successfully. Refreshing page...') # Notify user and refresh Streamlit
        st.rerun()

    # Check if the models folder exists
    if not os.path.isdir(MODEL_FOLDER):
        st.error('No models available. Please download models first.')
        st.stop() # Stop the app here

    # Load the models folder
    model_file_list = os.listdir(MODEL_FOLDER) # List all files inside the model folder
    model_names = [model_filename.split('.')[0] for model_filename in model_file_list if model_filename.endswith('.keras')] # Only keep model filenames ending with ".keras" and remove extension
    model_names = sorted(model_names) # Sort the model names alphabetically

    # Check if we have models in the folder
    if len(model_names) == 0:  # If no models found, stop the app
        st.error('No models available. Please download models first.')
        st.stop()

    # Let the user select one or preset it with the last selected one
    selected_model_name = st.selectbox('Select a model for prediction:',
                                       model_names,
                                       key='model_name')
    if selected_model_name is None: # If nothing is selected, stop the app
        st.stop()

    # Load the selected model
    model = ts.keras.models.load_model(os.path.join(MODEL_FOLDER, selected_model_name + '.keras'))
    st.write('Model loaded successfully.')
    st.write('Model summary:')
    df = pred.model_summary_to_df(model)
    st.dataframe(df, use_container_width=False)

    # Save it in streamlit session state
    classes = classes_2 if '2-classes' in selected_model_name else classes_4
    st.session_state['model'] = model
    st.session_state['classes'] = classes
    st.session_state['selected_model'] = selected_model_name

elif page == 'Prediction': # If the user navigates to the "Prediction" page

    # First check if we have models loaded
    if ('model' not in st.session_state): # Check if a model is loaded before making predictions
        st.error('No model loaded. Please select a model first.')
        st.stop() # Stop execution if no model is loaded

    model_name = st.session_state['selected_model'] # Load model and class list from Streamlit's session state
    model = st.session_state['model']
    classes = st.session_state['classes']

    # Let the user choose between file upload or URL input
    # Initialize variables
    image = None # To store the loaded image
    loading_type = 0 # 0 = nothing yet, 1 = from URL, 2 = from file upload
    st.markdown(content.prediction_note) # Display a helpful prediction note (markdown block from content file)
    url_input, divider, file_input = st.columns([3, 1, 3]) # Create three columns for layout: one for URL input, one as divider, one for file upload

    with divider: # Write the "or" divider between URL and file input
        st.markdown(content.prediction_or, unsafe_allow_html=True)

    with url_input: # If user uses the URL input field
        st.write('Enter a URL of an X-ray image for prediction:')
        # Text field where the user can paste an image URL (with a default example link)
        image_url = st.text_input('Image URL:', value='https://content.ca.healthwise.net/resources/14.1/en-ca/media/medical/hw/h9991297_001.jpg')
        if image_url and image_url != '': # If a URL is given
            image = pred.load_image_from_url(image_url) # Load image from the web
            st.write(f'You entered: {image_url}') # Show the entered URL
            loading_type = 1 # Mark that image came from URL

    with file_input: # Section to upload an image file instead of a URL
        st.write('Upload an X-ray image for prediction:')
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png']) # User uploads a file (only images: .jpg, .jpeg, .png allowed)
        if uploaded_file:
            image = pred.load_image_from_file(uploaded_file) # If an image file was uploaded, load it
            st.write(f'You uploaded: {uploaded_file.name}') # Confirm upload
            loading_type = 2 # Flag that the image came from file

    # Show the original image
    if image is None: # If no image was loaded from either method, stop the app
        st.stop()

    # Predict using the model
    # Show which model will be used and where the image came from
    if loading_type == 1:
        st.write('Predicting the following image from URL with model:', model_name)
    elif loading_type == 2:
        st.write('Predicting the following image from file with model:', model_name)

    # Show the loaded image
    st.image(image, width=300) # Show the image on screen (resized width to 300px for clarity)

    # Prepare the image for prediction in the selected model
    img_prepared = pred.prepare_image_for_model(image, model_name, model) 

    # Predict the prepared image and show the result as table
    pred_df = pred.predict_image(img_prepared, model_name, model, classes) # This returns a table with prediction probabilities
    st.dataframe(pred_df, use_container_width=False) # Show the prediction table nicely formatted

    # Optionally show a Grad-CAM
    st.write('Check the following checkbox to show a Grad-CAM of the prediction:') # Optional: checkbox to show Grad-CAM visualization
    show_gradcam = st.checkbox('Show Grad-CAM')
    if show_gradcam:
        pred.show_feature_maps(img_prepared, model_name, model) # Show feature maps using Grad-CAM

elif page == 'About':

    st.markdown(content.about)
