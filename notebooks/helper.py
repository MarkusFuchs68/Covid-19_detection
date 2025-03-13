import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


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

