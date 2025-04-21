import unittest
from noMLProc import identify_letter, calculate_distance, is_thumb_extended, count_extended_fingers

class TestASLFunctions(unittest.TestCase):

    def setUp(self):
        self.mock_landmarks_B = [
            (0.5, 0.8), #Wrist
            (0.45, 0.7),  # Thumb CMC
            (0.45, 0.6), #Thumb MCP: 
            (0.45, 0.5), # Thumb IP
            (0.45, 0.4),  # Thumb tip
            (0.55, 0.7),   # Index MCP
            (0.55, 0.5),  # Index PIP
            (0.55, 0.3),  # Index DIP
            (0.55, 0.2),  # Index tip
            (0.6, 0.7), # Middle MCP
            (0.6, 0.5),  # Middle PIP
            (0.6, 0.3),  # Middle DIP
            (0.6, 0.2),   # Middle tip
            (0.65, 0.7),   # Ring MCP
            (0.65, 0.5),  # Ring PIP
            (0.65, 0.3),  # Ring DIP
            (0.65, 0.2),  # Ring tip
            (0.7, 0.7), # Pinky MCP
            (0.7, 0.5),  # Pinky PIP
            (0.7, 0.3),   # Pinky DIP
            (0.7, 0.2),   # Pinky tip

        ]

    def test_identify_letter_B(self):
        """Test the identification of the letter 'B'."""
        print("Test landmarks:", self.mock_landmarks_B)
        result = identify_letter(self.mock_landmarks_B)
        print("Result:", result)
        self.assertEqual(result, 'B', f"Expected 'B', but got {result}")
    

if __name__ == '__main__':
    unittest.main()
