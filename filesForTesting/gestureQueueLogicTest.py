# A unit test for the queue data structure

import unittest
from queue import Queue

class testGestureQueue(unittest.TestCase):
    def setUp(self):
        """Set up a fresh queue for each test"""
        self.gestureQueue = Queue()

    def enqeueuAndDequeueTest(self):
        """Test if elements are enqueued and dequeued in FIFO order"""
        self.gestureQueue.put("A")
        self.gestureQueue.put("B")
        self.gestureQueue.put("C")

        self.assertEqual(self.gestureQueue.get(), "A")
        self.assertEqual(self.gestureQueue.get(), "B")
        self.assertEqual(self.gestureQueue.get(), "C")

    def emptyQueueTest(self):
        """Test behavior when dequeuing from an empty queue"""
        with self.assertRaises(Exception):  # Default Queue raises Exception when empty
            self.gestureQueue.get_nowait()

    def clearingTheQueue(self):
        """Test if the queue can be cleared correctly"""
        self.gestureQueue.put("A")
        self.gestureQueue.put("B")
        self.gestureQueue.queue.clear()
        self.assertEqual(len(self.gestureQueue.queue), 0)

if __name__ == "__main__":
    unittest.main()

    