import unittest
from noMLProc import identify_letter, calculate_distance, is_thumb_extended, count_extended_fingers

class TestASLFunctions(unittest.TestCase):

    def setUp(self):
        self.mock_landmarks_E = [
            (0.5, 0.7),  # Wrist
            (0.58, 0.58), # Thumb CMC
            (0.6, 0.4), # Thumb MCP
            (0.5, 0.3), # Thumb IP
            (0.42, 0.28), # Thumb Tip
            (0.6, 0.3), # Index MCP
            (0.6, 0.1), # Index PIP
            (0.58, 0.1), # Index DIP
            (0.57, 0.25), # Index Tip
            (0.5, 0.28),   # Middle MCP
            (0.5, 0.08),   # Middle PIP
            (0.6, 0.1),   # Middle DIP
            (0.5, 0.23),   # Middle Tip
            (0.45, 0.3),  # Ring MCP
            (0.43, 0.1),  # Ring PIP
            (0.44, 0.13),  # Ring DIP
            (0.45, 0.23),  # Ring Tip
            (0.39, 0.3),   # Pinky MCP
            (0.37, 0.17),   # Pinky PIP
            (0.4, 0.17),   # Pinky DIP
            (0.4, 0.24)    # Pinky Tip
        ]


    def test_identify_letter_E(self):
        """Test the identification of the letter 'E'."""
        print("Test landmarks:", self.mock_landmarks_E)
        result = identify_letter(self.mock_landmarks_E)
        print("Result:", result)
        self.assertEqual(result, 'E', f"Expected 'E', but got {result}")


if __name__ == '__main__':
    unittest.main()
