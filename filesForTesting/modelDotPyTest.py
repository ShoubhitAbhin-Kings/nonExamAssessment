# Test script for model.py

import unittest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import modelCreationAndTraining.model as model_script  

class TestModelDefinition(unittest.TestCase):

    def setUp(self):
        self.model = model_script.build_model()
        self.layers = self.model.layers

    def test_returnsSequentialModel(self):
        """Requirement: Defines a CNN architecture"""
        self.assertIsInstance(self.model, Sequential, "Model should be a Keras Sequential model")

    def test_containsMultipleConvLayers(self):
        """Requirement: Includes multiple convolutional layers"""
        conv_layers = [layer for layer in self.layers if isinstance(layer, Conv2D)]
        self.assertGreaterEqual(len(conv_layers), 3, "Model should have at least 3 Conv2D layers")

    def test_containsFlatteningLayer(self):
        """Requirement: Includes a flattening layer"""
        flatten_exists = any(isinstance(layer, Flatten) for layer in self.layers)
        self.assertTrue(flatten_exists, "Model should include a Flatten layer")

    def test_containsDenseLayer(self):
        """Requirement: Includes at least one fully connected Dense layer"""
        dense_layers = [layer for layer in self.layers if isinstance(layer, Dense)]
        self.assertGreaterEqual(len(dense_layers), 1, "Model should include at least one Dense layer")

    def test_containsDropout(self):
        """Requirement: Dropout to reduce overfitting"""
        dropout_exists = any(isinstance(layer, Dropout) for layer in self.layers)
        self.assertTrue(dropout_exists, "Model should include a Dropout layer")

    def test_outputLayerSoftmax(self):
        """Requirement: Output layer with softmax for classification"""
        output_layer = self.layers[-1]
        self.assertIsInstance(output_layer, Dense, "Output layer should be a Dense layer")
        self.assertEqual(output_layer.activation.__name__, 'softmax', "Output activation should be softmax")

    def test_modelIsCompiledCorrectly(self):
        """Requirement: Compiled with appropriate loss and metric"""
        self.assertEqual(self.model.loss, 'categorical_crossentropy', "Loss function should be categorical_crossentropy")
        self.assertIn('accuracy', self.model.metrics_names, "Model should include accuracy as a metric")

if __name__ == '__main__':
    unittest.main()