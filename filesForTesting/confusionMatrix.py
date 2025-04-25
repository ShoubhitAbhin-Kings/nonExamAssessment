import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

modelToBeLoaded = '/Users/shoubhitabhin/Documents/vsCodeProjects/ALevel/ALevelNEA/JanMMLV3/CNNModels/model6.keras'
dataDir = '/Users/shoubhitabhin/Documents/vsCodeProjects/ALevel/ALevelNEA/JanMMLV3/savedDataForAllModels/dataForsign_language_model/trainOnThese'

def generateConfusionMatrix(model, validationDatagen, dataDir, batchSize):
    """
    Generate and plot a confusion matrix for the model's predictions on the validation set.
    """
    # Prepare validation data generator
    valueGenerator = validationDatagen.flow_from_directory(
        os.path.join(dataDir, 'evaluation'),
        target_size=(300, 300),
        batch_size=batchSize,
        class_mode='categorical',
        shuffle=False  # Ensure that the order is kept to match true labels and predictions
    )

    # Get true labels (y_true) and predictions (y_pred)
    yTrue = valueGenerator.classes
    yPred = model.predict(valueGenerator, steps=len(valueGenerator), verbose=1)
    yPred = np.argmax(yPred, axis=1)  # Convert predicted probabilities to class indices

    # Compute confusion matrix
    cm = confusion_matrix(yTrue, yPred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valueGenerator.class_indices.keys(), yticklabels=valueGenerator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def loadTheModelAndGenerateAConfusionMatrix(modelPath, dataDir, batchSize=32):
    """
    Load a trained model and generate a confusion matrix for it.
    """
    # Load the trained model in H5 format
    model = tf.keras.models.load_model(modelToBeLoaded)

    # Initialize the validation data generator
    validationDatagen = ImageDataGenerator(rescale=1./255)

    # Generate confusion matrix for the model
    generateConfusionMatrix(model, validationDatagen, dataDir, batchSize)

if __name__ == "__main__":
    # Specify the path to the trained model and data directory used
    modelPath = modelToBeLoaded
    dataDir = dataDir
    
    loadTheModelAndGenerateAConfusionMatrix(modelPath, dataDir)