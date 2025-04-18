import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
from collections import deque  # For the queue functionality
import datetime  # For timestamping the file name
import time  # For managing the delay
import os # Allows choosing the model


class modelSelector:
    def __init__(self, modelDirectory):
        self.modelDirectory = modelDirectory

    def listTheModels(self):
        """Retursn a list of available model filenames from the directory"""
        return [f for f in os.listdir(self.modelDirectory) if f.endswith(".keras")]

    def selectAModel(self):
        """Prompts the user to select a model and return its' full path"""
        models = self.listTheModels()
        if not models:
            raise Exception("No .keras models found in the specified directory.")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        while True:
            try:
                choice = int(input("Select model number: ")) - 1
                if 0 <= choice < len(models):
                    return os.path.join(self.modelDirectory, models[choice])
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")

    def loadModel(self, modelPath):
        """Load and return the selected model"""
        return load_model(modelPath)
    
# Initialize model selector and load model
try:
    """bug fix, assign an instance of the modelSelector class to the variable modelSelectorInstance to make it clear that we must have an instance of the class named differently to the class itself"""
    modelSelectorInstance = modelSelector("CNNModels") 
    selectedModelPath = modelSelectorInstance.selectAModel()
    model = modelSelectorInstance.loadModel(selectedModelPath)
    print(f"Loaded model: {selectedModelPath}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define class labels (ensure these match the training class order, so when more letters are added make sure they match)
classLabels = ['A', 'B', 'C', 'D', 'E', 'Unknown']

class gestureQueue:
    def __init__(self, maxSize: int):
        self.queue = deque(maxlen=maxSize)  # Use deque for efficient FIFO operations
        self.maxSize = maxSize

    def enqueue(self, letter: str):
        """Add a letter to the queue"""
        self.queue.append(letter)

    def dequeue(self):
        """Remove a letter from the queue"""
        if self.queue:
            return self.queue.popleft()
        return None

    def clear(self):
        """Clear the entire queue"""
        self.queue.clear()

    def getQueue(self):
        """Return the list of letters in the queue"""
        return list(self.queue)
    

# Define SignLanguageTranslatorUI class
class signLanguageTranslatorApp:
    def __init__(self, windowTitle: str, quitMessage: str):
        self.windowTitle = windowTitle
        self.quitMessage = quitMessage
        self.classLabels = classLabels
        self.predicted_label = ""
        self.predictions = []
        self.gestureQueue = gestureQueue(maxSize=10)  # Initialise the gesture queue with a max size of 10
        self.lastPredictedLabel = None  # Tracks the last predicted letter added to the queue
        self.stableFrameCount = 0  # Counter to track how many frames the prediction has been stable
        self.stableFrameThreshold = 10  # Number of frames the prediction must remain stable before being enqueued
        self.statusMessage = ""  # For displaying status messages on the screen
        self.statusTimestamp = None  # Timestamp when the status message is shown
        self.confidenceThreshold = 0.90 # Initialises the confidence threshold

        # Initialise the OpenCV window
        cv2.namedWindow(self.windowTitle)

    def showTitle(self, img):
        """Show the title text at the top of the window"""
        cv2.putText(img, self.windowTitle, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def showQuitMessage(self, img):
        """Show the instructions to quit the program"""
        cv2.putText(img, self.quitMessage, (50, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def showPredictions(self, img):
        """Show the predicted label and probabilities on the image"""
        y_offset = 100  # Starting point for displaying predictions
        for i, (label, prob) in enumerate(zip(self.classLabels, self.predictions)):
            cv2.putText(img, f"{label}: {prob*100:.2f}%", (50, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def showGestureQueue(self, img):
        """Display the gesture queue on the screen"""
        y_offset = img.shape[0] - 100  # Starting point for displaying the queue
        queue_str = "Queue: " + "".join(self.gestureQueue.getQueue())  # Get all letters in the queue
        cv2.putText(img, queue_str, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def showStatusMessage(self, img):
        """Display a status message on the screen for 2 seconds"""
        yOffset = img.shape[0] - 150  # Starting point for displaying the message
        if self.statusMessage and self.statusTimestamp:
            elapsed_time = time.time() - self.statusTimestamp  # Calculate elapsed time
            if elapsed_time < 2:
                cv2.putText(img, self.statusMessage, (50, yOffset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                self.statusMessage = ""  # Clear message after 2 seconds
                self.statusTimestamp = None  # Reset timestamp

    def showError(self, img, message="Error occurred"):
        """Displays a temporary error message on the screen for 1 second"""
        # Draw the error message in red
        cv2.putText(img, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the frame with the error message
        cv2.imshow("Sign Language Translator", img)
        
        # Wait for however long I feel like rn
        cv2.waitKey(10)

    def updateDisplay(self, predicted_label, predictions, img):
        """Update the displayed information"""
        self.predicted_label = predicted_label
        self.predictions = predictions
        self.showTitle(img)
        self.showPredictions(img)
        self.showQuitMessage(img)
        self.showGestureQueue(img)  # Display the gesture queue
        self.showStatusMessage(img)  # Display status message

    def processPredictedLetter(self, predicted_label):
        """Add predicted letter to the queue if stable for the defined number of frames"""
        if predicted_label != self.lastPredictedLabel:
            self.stableFrameCount = 0  # Reset counter if the predicted label changes
            self.lastPredictedLabel = predicted_label
        else:
            self.stableFrameCount += 1

        if self.stableFrameCount >= self.stableFrameThreshold:
            if predicted_label != 'Unknown': 
                self.gestureQueue.enqueue(predicted_label)
            else: # If it's Unknown, it will add a space in the queue
                self.gestureQueue.enqueue('#')
            self.stableFrameCount = 0  # Reset after enqueuing

    def saveQueueToFile(self):
        """Save the current gesture queue to a text file with a timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sequenceOfLetters{timestamp}.txt"
        with open(filename, 'w') as f:
            # Join the queue elements into a single string and write it to the file
            f.write("".join(self.gestureQueue.getQueue()))
        print(f"Queue saved to {filename}")
        self.statusMessage = f"Queue saved as {filename}"  # Update the status message
        self.statusTimestamp = time.time()  # Set the timestamp for the message


"""Start of the main program logic"""
# Initialise webcam and hand detector

try:
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    if not cap.isOpened():
        raise Exception("Webcam not accessible. Please check your connection.")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit(1)  # Exit if the webcam cannot be accessed

# Initialize the UI class
ui = signLanguageTranslatorApp("Sign Language Translator", "Press Q to Quit | Press C to Clear Queue | Press S to Save Queue")

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        try:
            # Extract and preprocess the hand region
            hand_img = img[y:y+h, x:x+w]
            hand_img = cv2.resize(hand_img, (300, 300))
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            hand_img = hand_img / 255.0  # Normalize
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict the sign language letter
            prediction = model.predict(hand_img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)  # Get the maximum confidence for the predicted class

            if confidence < ui.confidenceThreshold:
                predicted_label = 'Unknown'  # If confidence is too low, classify as Unknown
            else:
                predicted_label = classLabels[predicted_class]  # Otherwise, classify normally

            probabilities = prediction[0]  # Get probabilities for all classes

            # Update the UI with predictions and process the predicted letter
            ui.processPredictedLetter(predicted_label)  # Check if letter should be enqueued
            ui.updateDisplay(predicted_label, probabilities, img)
        except Exception as e:
            print(f"Bring your hand in the frame: {e}")
            ui.showError(img, "Bring your hand in the frame")

    # Get the image width and height
    height, width, _ = img.shape

    # Define the position for the top right (adjust 20 pixels from the right edge)
    x_pos = width - 500  # You can adjust the 250 value to fit the text nicely
    y_pos = 50  # Keep it at the top of the window

    # Place the Confidence Threshold text at the top right
    cv2.putText(img, f"Confidence Threshold: {ui.confidenceThreshold:.2f}", 
                (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Keyboard controls for adjusting confidence threshold
    key = cv2.waitKey(1) & 0xFF
    if key == ord('='):  # Increase threshold - PRESS THE +/= key but CV2 recognises it as the = key
        ui.confidenceThreshold = min(1.0, ui.confidenceThreshold + 0.05)  # Max threshold is 1.0
        print(f"Threshold increased: {ui.confidenceThreshold}")
    elif key == ord('-'):  # Decrease threshold
        ui.confidenceThreshold = max(0.0, ui.confidenceThreshold - 0.05)  # Min threshold is 0.0
        print(f"Threshold decreased: {ui.confidenceThreshold}")

    # Show the image with updated UI
    cv2.imshow("Sign Language Recognition", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Clear the queue on 'c' key press
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ui.gestureQueue.clear()
        ui.statusMessage = "Queue cleared"  # Display message on screen
        ui.statusTimestamp = time.time()  # Set timestamp for the message
    
    # Save the queue to a text file on 's' key press
    if cv2.waitKey(1) & 0xFF == ord('s'):
        ui.saveQueueToFile()
        ui.statusMessage = "File saved"  # Display message on screen
        ui.statusTimestamp = time.time()

cap.release()
cv2.destroyAllWindows()