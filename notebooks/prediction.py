# Import standard Python and scientific libraries
import os # for file path operations
import numpy as np # for numerical operations
import pandas as pd # for data table handling
import urllib # to load images from URLs
import cv2 # for image decoding (OpenCV)
import matplotlib.pyplot as plt # for plotting

import tensorflow as tf # Import TensorFlow and Keras for deep learning
from tensorflow.keras.models import Model # to load and use models
from tensorflow.keras.layers import Conv2D # to work with convolution layers


# Note: the predefined class-names are the usual order, when reading in the dataset, but be careful!
def load_image_from_filepath(file_path):
    # Read the image as raw binary data
    img_raw = tf.io.read_file(file_path)

    # Split the file extension (e.g., ".jpg", ".png")
    _, ext = os.path.splitext(file_path)
    img_decoded = None
    if ext == '.jpg' or ext == '.jpeg': # If the file is a JPEG image
        img_decoded = tf.image.decode_jpeg(img_raw, channels = 3)
    elif ext == '.png':  # If the file is a PNG image
        img_decoded = tf.image.decode_png(img_raw, channels = 3)
    else:
        raise RuntimeError(f'File extension {ext} not supported.') # If it's another format, raise an error
    
    return img_decoded.numpy() # Convert Tensor to NumPy array


def load_image_from_url(url):  # Open the URL and read the bytes
    resp = urllib.request.urlopen(url)
    img_raw = np.asarray(bytearray(resp.read()), dtype='uint8')
    img_decoded = cv2.imdecode(img_raw, -1) # Decode the image using OpenCV
    return img_decoded


def show_predict_image(img_name, img_decoded, models, class_names=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']):
    
    # Plot the image (scaled between 0 and 1 for display)
    plt.figure(figsize=(3,3))
    plt.imshow(img_decoded / 255)
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name)
    plt.show()

    # Prepare a DataFrame to store prediction results
    model_pred = 'Model_Prediction'
    pred_df = pd.DataFrame(columns=[model_pred] + class_names)

    # Predict using every model in the dictionary
    for model_name, model in models.items():

        # Copy the image to avoid overwriting
        img = img_decoded.copy() # Copy the image to avoid modifying the original

        # Make grayscale if model requires it
        print('Model input shape is:', model.input_shape)
        if model.input_shape[-1] == 1: # If the model expects grayscale images (1 channel), convert it
            print(f'Converting image to grayscale for model {model_name}')
            img = tf.image.rgb_to_grayscale(img) if img.shape[-1] == 3 else img   # Convert to grayscale if it’s a color image (3 channels)

        # Resize according to the models input shape
        print(f'Resizing for model {model_name} to input_shape: {model.input_shape[1:3]}') # Resize the image to match what the model expects (e.g., 224x224)
        img = tf.image.resize(img, size=model.input_shape[1:3])

        # Run the prediction using the model
        # Note: We wrap img in a list to simulate a batch of 1 image
        pred = model.predict(np.array([img], dtype=np.float32))[0]

        # Store prediction results in a table
        # - First column: predicted class label (the one with highest probability)
        # - Following columns: class probabilities rounded to 3 decimal points
        pred_df.loc[model_name] = [class_names[np.argmax(pred)]] + pred.round(3).tolist()

    # Show the prediction report
    display(pred_df) # Show the full prediction table
