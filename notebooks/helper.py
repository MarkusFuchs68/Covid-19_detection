import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D 


def plot_learning_curve(history_model, loss='loss', metric='accuracy'):
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(history_model.history[loss])
    plt.plot(history_model.history[str('val_') + loss])
    plt.title(f'Model {loss} by epoch')
    plt.ylabel(loss)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(history_model.history[metric])
    plt.plot(history_model.history[str('val_') + metric])
    plt.title(f'Model {metric} by epoch')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.show()


# Get true labels and predictions from the test dataset
def get_predictions_and_labels(model, dataset):
    true_labels = []
    pred_labels = []

    for images, labels in dataset:

        # Get the model's predictions
        preds = model.predict(images, verbose=0)
        # Get the predicted labels (argmax)
        pred_labels.extend(np.argmax(preds, axis=-1))

        true_labels.extend(labels.numpy())  # Get the true labels

    return np.array(true_labels), np.array(pred_labels)


# Print a report with classification_report and heatmap confusion_matrix
def report_model_performance(y_true, y_pred, class_names):
    # Print the classification report (precision, recall, F1-score)
    cr = classification_report_imbalanced(y_true, y_pred, target_names=class_names, output_dict=True)
    # Make the report a pandas array
    df_cr = pd.DataFrame(cr).transpose()
    display(df_cr)

    # Show also the non-normalized crosstab
    # display(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']))
    ct = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
    column_mapping = {index: class_name for index, class_name in enumerate(class_names)}
    ct = ct.rename(columns=column_mapping)
    ct.index = class_names
    display(ct)

    # Display the confusion matrix
    plt.figure(figsize=(4, 4))
    # Compute the normalized confusion matrix (normalized on columns -> we get recall so)
    cnf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    # Plot the confusion matrix as a heatmap
    sns.heatmap(cnf_matrix, cmap='Blues', annot=True, cbar=False, fmt=".2f")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=np.arange(0.5, len(class_names)+0.5, 1),
               labels=class_names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(0.5, len(class_names)+0.5, 1),
               labels=class_names, rotation=45, ha='right')
    plt.show()

    # Return a numpy array, from where we can copy and paste the values
    # into an evaluation excel sheet
    report = {}
    report['crosstab'] = ct
    report['classification_report'] = df_cr
    report['confusion_matrix'] = cnf_matrix
    return report


# Function to extract model summary as a DataFrame
def model_summary_to_df(model):
    import io

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
            param_count = parts[-1]
            data.append([layer_nametype, output_shape, param_count])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Layer Name (type)", "Output Shape", "Param Count"])
    return df


def show_feature_maps(model, image):
    """
    Displays how the input image (Shape: (224, 224, 3), pixel-value-range 0-255),
    makes its way through the convolutional layers
    """
    # Retrieve the names of all (even recursive) convolution layers in the model
    def get_conv_layers(model):
        inner_conv_layers = []
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                inner_conv_layers.append((model,layer.name))
            elif isinstance(layer, Model):
                inner_conv_layers.extend(get_conv_layers(layer))
        return inner_conv_layers

    conv_layers = get_conv_layers(model)

    # Loop through all convolution layers
    for j, (inner_model,layer) in enumerate(conv_layers):

        # Create a new model with the same input as the original model but with the output
        # of the specific convolution layer
        conv_model = Model(inputs=inner_model.input, outputs=inner_model.get_layer(layer).output)

        # Add an extra dimension to the image to make it compatible with the batch (shape: (1, H, W, C))
        image_batch = tf.expand_dims(image, axis=0)

        # Normalize (optional, depends on model)
        image_batch = image_batch / 255.0  # Normalize to [0,1]

        # Predict the feature maps for the given image using the created model
        feature_maps = conv_model.predict(image_batch, verbose=0)

        # Squeeze to remove unnecessary dimensions, resulting in an array of shape (H, W, N)
        feature_maps = tf.squeeze(feature_maps)

        # Initialize a figure to display the feature maps
        plt.figure(figsize=(12, 12))

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
            plt.imshow(feature_maps[..., i])  # Display the feature map
            plt.axis("off")  # Turn off the axes
            plt.title(f'Output {layer} filter {i+1}', fontsize=16 - nb_subplot - 1)  # Add a title

    # Display the results
    plt.show()