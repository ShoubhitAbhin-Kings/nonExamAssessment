# Test script for train.py
# PLEASENOTE - This script was heavily written by CHATGPT

import unittest
from unittest.mock import patch, MagicMock
from modelCreationAndTraining.train import signLanguageTrainer

class TestSignLanguageTrainer(unittest.TestCase):

    @patch('modelCreationAndTraining.train.EarlyStopping')
    @patch('modelCreationAndTraining.train.ImageDataGenerator')
    @patch('modelCreationAndTraining.train.build_model')
    @patch('modelCreationAndTraining.train.compute_class_weight')
    def test_training_process(
        self, mock_compute_class_weight, mock_build_model, mock_ImageDataGen, mock_EarlyStopping
    ):
        # Setup fake components
        fake_train_generator = MagicMock()
        fake_val_generator = MagicMock()

        fake_train_generator.samples = 64
        fake_val_generator.samples = 16
        fake_train_generator.classes = [0, 1, 0, 1] * 8
        fake_train_generator.class_indices = {'A': 0, 'B': 1}

        fake_val_generator.classes = [0, 1] * 8
        fake_val_generator.class_indices = {'A': 0, 'B': 1}

        mock_gen_instance = MagicMock()
        mock_gen_instance.flow_from_directory.side_effect = [fake_train_generator, fake_val_generator]
        mock_ImageDataGen.return_value = mock_gen_instance

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        mock_compute_class_weight.return_value = [1.0, 1.5]

        # Instantiate trainer
        trainer = signLanguageTrainer(data_dir='fake_dir', batch_size=8, epochs=1)
        trainer.train()

        # Requirement: buildModel() is called with correct shape
        mock_build_model.assert_called_once_with(input_shape=(300, 300, 3), num_classes=2)

        # Requirement: class weights are calculated
        mock_compute_class_weight.assert_called_once()

        # Requirement: EarlyStopping is initialized with correct monitor
        mock_EarlyStopping.assert_called_once()
        args, kwargs = mock_EarlyStopping.call_args
        self.assertEqual(kwargs['monitor'], 'val_accuracy')
        self.assertEqual(kwargs['restore_best_weights'], True)

        # Requirement: model.fit is called with correct generators
        mock_model.fit.assert_called_once()
        fit_args, fit_kwargs = mock_model.fit.call_args
        self.assertEqual(fit_kwargs['class_weight'], {0: 1.0, 1: 1.5})
        self.assertEqual(fit_kwargs['validation_data'], fake_val_generator)

        # Requirement: model is saved
        mock_model.save.assert_called_once()
        print("✅ All model training requirements were logically tested.")

if __name__ == "__main__":
    unittest.main()