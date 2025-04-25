# Unit test for the dataCollection.py script

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import dataCollectionAndAugmentation.dataCollection as dc
import cv2

class TestDataCollection(unittest.TestCase):

    def setUp(self):
        self.imageSize = 300
        self.offset = 30
        self.folder = 'testData'
        os.makedirs(self.folder, exist_ok=True)

    def tearDown(self):
        # Clean up after tests
        for file in os.listdir(self.folder):
            os.remove(os.path.join(self.folder, file))
        os.rmdir(self.folder)

    @patch('cv2.VideoCapture')
    def test_initialiseCamera(self, mock_video):
        cap_mock = MagicMock()
        mock_video.return_value = cap_mock
        cap, detector = dc.initialiseCamera()
        self.assertIsNotNone(cap)
        self.assertIsNotNone(detector)

    def test_resizeTallImages(self):
        img = np.ones((400, 200, 3), dtype=np.uint8) * 100
        result = dc.resizeImage(img, self.imageSize, 400, 200)
        self.assertEqual(result.shape, (self.imageSize, self.imageSize, 3))

    def test_resizeWideImages(self):
        img = np.ones((200, 400, 3), dtype=np.uint8) * 100
        result = dc.resizeImage(img, self.imageSize, 200, 400)
        self.assertEqual(result.shape, (self.imageSize, self.imageSize, 3))

    def test_saveImage(self):
        dummy = np.ones((300, 300, 3), dtype=np.uint8) * 123
        counter = 0
        dc.saveImage(dummy, self.folder, counter)
        files = os.listdir(self.folder)
        self.assertGreater(len(files), 0)
        self.assertTrue(files[0].endswith('.jpg'))

    @patch('dataCollectionAndAugmentation.dataCollection.HandDetector')
    @patch('cv2.VideoCapture')
    def test_captureHandImageMocked(self, mock_video, mock_detector_class):
        # Simulate image capture
        cap_mock = MagicMock()
        cap_mock.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
        mock_video.return_value = cap_mock

        detector_instance = MagicMock()
        detector_instance.findHands.return_value = (
            [{"bbox": [100, 100, 200, 200]}], np.ones((480, 640, 3), dtype=np.uint8)
        )
        mock_detector_class.return_value = detector_instance

        cap, detector = dc.initialiseCamera()
        crop, full = dc.captureHandImage(cap, detector, self.offset, self.imageSize)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[2], 3)  # Check it's a color image

if __name__ == "__main__":
    unittest.main()