import io
from altair import param
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

import streamlit as st


# Function to extract model summary as a DataFrame
def model_summary_to_df(model):

    # Redirect sys.stdout to capture model.summary() output
    stream = io.StringIO()

    # Redirect Keras summary output to a variable
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()

    # Parse summary output
    lines = summary_str.split("\n")
    data = []
    for line in lines[2:-4]:  # Skip headers and footer
        parts = [x for x in line.split("│") if x]  # Split by │ and remove empty elements
        if len(parts) >= 3:
            layer_nametype = parts[0]
            output_shape = parts[1]
            param_count = parts[2]
            if (len(layer_nametype.strip()) > 0 or len(output_shape.strip()) > 0 or len(param_count.strip()) > 0):
                data.append([layer_nametype, output_shape, param_count])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Layer Name (type)", "Output Shape", "Param Count"])
    return df


def load_image_from_file(uploaded_file):
    # Load file as raw bytes
    image_bytes = uploaded_file.read()
    # Decode the image into a grayscale tensor
    try:
        image = tf.image.decode_image(image_bytes, channels=1)
    except Exception as e:
        st.error(f"Error during decoding the image: {e}")
        return None
    return image.numpy()


def load_image_from_url(url):
    # Load file as raw bytes
    try:
        response = requests.get(url)
    except Exception:
        st.error(f"Failed to load image from URL. Please check proper URL.")
        return None
    image_bytes = response.content
    # Decode the image into a grayscale tensor
    try:
        image = tf.image.decode_image(image_bytes, channels=1)
    except Exception as e:
        st.error(f"Error during decoding the image: {e}")
        return None
    return image.numpy()


def prepare_image_for_model(image, model_name, model, normalize = False):
    # Copy the image to avoid overwriting
    img_tf = tf.convert_to_tensor(image.copy())
    channels = model.input_shape[-1]
    target_size = model.input_shape[1:3]

    # Ensure 3D shape (H, W, C)
    if img_tf.ndim == 2:  # Grayscale (H, W)
        img_tf = tf.expand_dims(img_tf, axis=-1)  # (H, W, 1)

    # Convert grayscale to RGB if needed
    if img_tf.shape[-1] == 1 and channels == 3:
        img_tf = tf.image.grayscale_to_rgb(img_tf)
    elif img_tf.shape[-1] == 3 and channels == 1:
        img_tf = tf.image.rgb_to_grayscale(img_tf)

    # Resize
    img_tf = tf.image.resize(img_tf, target_size)

    # Normalize
    if normalize:
        img_tf = img_tf / 255.0

    return img_tf


# Note: the predefined class-names are the usual order, when reading in the dataset, but be careful!
def predict_image(img_prepared, model_name, model, class_names):

    # Prepare the prediction report
    model_pred = 'Predicted class'
    pred_df = pd.DataFrame(columns=[model_pred] + class_names)

    # Predict using the provided model
    img_batch = tf.expand_dims(img_prepared, axis=0)
    pred = model.predict(img_batch)[0]

    # Show the probabilities of the predicted classes
    pred_df.loc[model_name] = [class_names[np.argmax(pred)]] + pred.round(3).tolist()
    pred_df.index.name = 'Model'

    # Return the prediction report
    return pred_df


def show_feature_maps(img_prepared, model_name, model):

    # Retrieve the names of all (even recursive) convolution layers in the model
    def get_conv_layers(model):
        inner_conv_layers = []
        for layer in model.layers:
            # We need to srict on real conv layers (not reduce, pooling, etc.)
            if isinstance(layer, Conv2D) and 'conv' in str(layer.name.lower()):
                st.write(f'Found convolution layer: {layer.name}')
                inner_conv_layers.append((model, layer.name))
            elif isinstance(layer, Model):
                st.write(f'Found inner model layer: {layer.name}')
                inner_conv_layers.extend(get_conv_layers(layer))

        return inner_conv_layers

    # Get all convolution layers in the model, including inner models
    conv_layers = get_conv_layers(model)
    if len(conv_layers) == 0:
        st.error('No convolution layers found in the model.')
        return

    # Get first layer from the model as our input definition
    try:  # Remember, the list holds tuples of model and layer name
        # If this fails, then because our model input is not defined yet
        input = conv_layers[0][0].input # take it from the model itself
    except AttributeError:  # input is not yet defined
        # If it fails, we need to get the input from the first convolution layer
        input = conv_layers[0][0].get_layer(conv_layers[0][1]).input  # take it from the first layer of the model

    # Loop through all convolution layers
    for j, (inner_model, layer) in enumerate(conv_layers):

        # Create a new model with the same input as the original model but with the output
        conv_model = Model(inputs=input, outputs=inner_model.get_layer(layer).output)

        # Show the layer name
        st.write(f'Layer {j+1}/{len(conv_layers)}: {layer}')

        # Add an extra dimension to the image to make it compatible with the batch (shape: (1, H, W, C))
        image_batch = tf.expand_dims(img_prepared, axis=0)

        # Predict the feature maps for the given image using the created model
        feature_maps = conv_model.predict(image_batch, verbose=0)

        # Squeeze to remove unnecessary dimensions, resulting in an array of shape (H, W, N)
        feature_maps = tf.squeeze(feature_maps)

        # Initialize a figure to display the feature maps
        fig = plt.figure(figsize=(12, 12))

        # Loop through all feature maps of the layer
        for i in range(feature_maps.shape[-1]):

            # Calculate the number of subplots needed to display all feature maps
            nb_subplot = feature_maps.shape[-1]**(1/2)

            # If the number of subplots is not an integer, round up to the next integer
            if nb_subplot - int(nb_subplot) != 0:
                nb_subplot = int(nb_subplot) + 1
            else: 
                nb_subplot = int(nb_subplot)

            # Create a subplot for each feature map
            plt.subplot(nb_subplot, nb_subplot, i + 1)
            plt.imshow(feature_maps[..., i], cmap='viridis')  # Display the feature map
            plt.axis("off")  # Turn off the axes
            plt.title(f'Output {layer} filter {i+1}', fontsize=16 - nb_subplot - 1)  # Add a title

        # Show the figure with all feature maps
        st.pyplot(fig)
        plt.close(fig)
