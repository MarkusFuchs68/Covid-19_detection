import os
import numpy as np
import pandas as pd
import urllib
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D


# Note: the predefined class-names are the usual order, when reading in the dataset, but be careful!
def load_image_from_filepath(file_path):
    # Read the raw image
    img_raw = tf.io.read_file(file_path)

    # Check extension
    _, ext = os.path.splitext(file_path)
    img_decoded = None
    if ext == '.jpg' or ext == '.jpeg':
        img_decoded = tf.image.decode_jpeg(img_raw, channels = 3)
    elif ext == '.png':
        img_decoded = tf.image.decode_png(img_raw, channels = 3)
    else:
        raise RuntimeError(f'File extension {ext} not supported.')
    
    return img_decoded.numpy()


def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    img_raw = np.asarray(bytearray(resp.read()), dtype='uint8')
    img_decoded = cv2.imdecode(img_raw, -1)
    return img_decoded


def show_predict_image(img_name, img_decoded, models, class_names=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']):
    
    # Show the image
    plt.figure(figsize=(3,3))
    plt.imshow(img_decoded / 255)
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name)
    plt.show()

    # Prepare the prediction report
    model_pred = 'Model_Prediction'
    pred_df = pd.DataFrame(columns=[model_pred] + class_names)

    # Predict with all models given
    for model_name, model in models.items():

        # Copy the image to avoid overwriting
        img = img_decoded.copy()

        # Make grayscale if model requires it
        print('Model input shape is:', model.input_shape)
        if model.input_shape[-1] == 1:
            print(f'Converting image to grayscale for model {model_name}')
            img = tf.image.rgb_to_grayscale(img) if img.shape[-1] == 3 else img  # Convert if needed

        # Resize according to the models input shape
        print(f'Resizing for model {model_name} to input_shape: {model.input_shape[1:3]}')
        img = tf.image.resize(img, size=model.input_shape[1:3])

        # Predict using the provided model
        pred = model.predict(np.array([img], dtype=np.float32))[0]

        # Show the probabilities of the predicted classes
        pred_df.loc[model_name] = [class_names[np.argmax(pred)]] + pred.round(3).tolist()

    # Show the prediction report
    display(pred_df)
