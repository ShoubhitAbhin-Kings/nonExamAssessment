# Test for the dataAugmentation.py script

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np
import cv2

# Patch os.listdir and cv2.imread
class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        # Fake image: 300x300 black image
        self.testImage = np.zeros((300, 300, 3), dtype=np.uint8)
        self.fakeFolder = 'A'
        self.imageName = 'test.jpg'
        self.imagePath = f'/fake_path/{self.imageName}'

    @patch('cv2.imwrite')
    @patch('cv2.imread')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_data_augmentation_logic(
        self, mockListDir, mockIsDir, mockExists, mock_makedirs, mockImRead, mockImWrite
    ):
        # Simulate directory structure
        mockListDir.side_effect = [
            [self.fakeFolder],  # top-level folder
            [self.imageName]    # inside 'A'
        ]
        mockIsDir.return_value = True
        mockExists.return_value = True
        mockImRead.return_value = self.testImage

        # Import the augmentation script
        import dataCollectionAndAugmentation.dataAugmentation

        # Check that images were attempted to be written
        self.assertTrue(mockImWrite.called)
        self.assertGreater(mockImWrite.call_count, 0)

    def tearDown(self):
        pass  # No clean-up needed as everything was mocked

if __name__ == "__main__":
    unittest.main()