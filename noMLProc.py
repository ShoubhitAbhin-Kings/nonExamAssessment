import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def preprocess_image(image_path):
    """Load and preprocess the image for landmark extraction."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return image, results

def extract_landmarks(results):
    """Extract hand landmarks and return a list of landmark points."""
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.append((point.x, point.y))
        return landmarks
    return None

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def is_finger_extended(landmarks, tip_index, dip_index, pip_index):
    """
    Check if a finger is extended using relative y-coordinates.
    The finger is extended if the tip is above both DIP and PIP.
    """
    return (landmarks[tip_index][1] < landmarks[dip_index][1] < landmarks[pip_index][1])

def is_thumb_extended(landmarks):
    """
    Enhanced thumb detection by measuring both horizontal spread and proximity to the palm.
    """
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    index_base = landmarks[5]

    tip_mcp_distance = calculate_distance(thumb_tip, thumb_mcp)
    tip_index_distance = calculate_distance(thumb_tip, index_base)

    def calculate_angle(a, b, c):
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        dot_product = ab[0] * cb[0] + ab[1] * cb[1]
        mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
        angle = math.acos(dot_product / (mag_ab * mag_cb + 1e-6))
        return math.degrees(angle)

    thumb_angle = calculate_angle(wrist, thumb_mcp, thumb_tip)

    if tip_mcp_distance > 0.2 and tip_index_distance > 0.2 and thumb_angle > 30:
        return True
    else:
        return False

def count_extended_fingers(landmarks):
    """Count the number of extended fingers excluding the thumb."""
    return sum([
        is_finger_extended(landmarks, 8, 6, 5),
        is_finger_extended(landmarks, 12, 10, 9),
        is_finger_extended(landmarks, 16, 14, 13),
        is_finger_extended(landmarks, 20, 18, 17)
    ])

def identify_letter(landmarks):
    """Identify ASL letters based on optimized geometric rules."""

    print("Processing landmarks:", landmarks)
    thumb_extended = is_thumb_extended(landmarks)
    extended_fingers = count_extended_fingers(landmarks)

    # A: Thumb extended slightly, all fingers folded
    if thumb_extended and extended_fingers == 0:
        return "A"
    # B: All fingers extended straight up, thumb folded across palm
    elif not thumb_extended and extended_fingers == 4:
        return "B"
    # C: All fingers curved forming a 'C' shape, measured using thumb to index distance
    elif thumb_extended and extended_fingers == 4:
        index_thumb_distance = calculate_distance(landmarks[4], landmarks[8])
        if index_thumb_distance > 0.1:
            return "C"
    # D: Index finger extended, thumb touching index forming circle
    elif extended_fingers == 1 and calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "D"
    # E: All fingers folded into the palm with thumb folded
    elif not thumb_extended and extended_fingers == 0:
        return "E"
    # F: Thumb and index touching, other fingers extended
    elif extended_fingers == 3 and calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "F"
    # G: Thumb and index extended horizontally
    elif extended_fingers == 1 and thumb_extended and landmarks[8][1] > landmarks[6][1]:
        return "G"
    # H: Index and middle extended horizontally, thumb folded
    elif extended_fingers == 2 and thumb_extended and landmarks[8][1] > landmarks[6][1]:
        return "H"
    # I: Only pinky extended
    elif extended_fingers == 1 and landmarks[20][1] < landmarks[18][1]:
        return "I"
    # J: Same as I, but motion-tracing (J motion not detectable here)
    elif extended_fingers == 1 and landmarks[20][1] < landmarks[18][1]:
        return "J"
    # K: Index and middle extended in a V shape with thumb extended
    elif extended_fingers == 2 and thumb_extended:
        return "K"
    # L: Thumb and index forming an L shape
    elif extended_fingers == 1 and thumb_extended:
        return "L"
    # M: Thumb under three folded fingers
    elif extended_fingers == 0 and calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "M"
    # N: Thumb under two folded fingers
    elif extended_fingers == 0 and calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "N"
    # O: Thumb and index forming a circular shape
    elif extended_fingers == 0 and calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "O"
    # P: Thumb and index touching, index pointing downwards
    elif extended_fingers == 1 and thumb_extended:
        return "P"
    # Q: Similar to G but pointing downwards
    elif extended_fingers == 1 and thumb_extended and landmarks[8][1] > landmarks[6][1]:
        return "Q"
    # R: Index and middle crossed
    elif extended_fingers == 2 and calculate_distance(landmarks[8], landmarks[12]) < 0.05:
        return "R"
    # S: Fist with thumb across fingers
    elif extended_fingers == 0 and not thumb_extended:
        return "S"
    # T: Thumb between index and middle
    elif extended_fingers == 0:
        return "T"
    # U: Index and middle extended together
    elif extended_fingers == 2 and calculate_distance(landmarks[8], landmarks[12]) > 0.05:
        return "U"
    # V: Index and middle forming a V shape
    elif extended_fingers == 2 and calculate_distance(landmarks[8], landmarks[12]) < 0.1:
        return "V"
    # W: Three fingers extended
    elif extended_fingers == 3:
        return "W"
    # X: Index bent, others folded
    elif extended_fingers == 1 and landmarks[8][1] > landmarks[6][1]:
        return "X"
    # Y: Thumb and pinky extended
    elif extended_fingers == 1 and landmarks[20][1] < landmarks[18][1]:
        return "Y"
    # Z: Index tracing a Z shape (motion not detected)
    return "Unknown"

def main(image_path):
    image, results = preprocess_image(image_path)
    landmarks = extract_landmarks(results)


    if landmarks:
        extended_fingers = count_extended_fingers(landmarks)
        thumb_extended = is_thumb_extended(landmarks)
        predicted_letter = identify_letter(landmarks)

        cv2.putText(image, f"Extended Fingers: {extended_fingers}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Thumb Extended: {'Yes' if thumb_extended else 'No'}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Predicted Letter: {predicted_letter}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, lm, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No hand detected.")

if __name__ == "__main__":
    image_path = "/Users/shoubhitabhin/Downloads/ASL/C-ASL.jpg"
    main(image_path)


