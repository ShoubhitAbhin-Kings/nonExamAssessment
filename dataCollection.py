import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialise webcam and hand detector
def initialiseCamera():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    return cap, detector

# Capture and process hand image
def captureHandImage(cap, detector, offset, imgSize):
    success, image = cap.read()
    if not success:
        return None, None
    
    hands, image = detector.findHands(image)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imageCrop = image[max(0, y - offset): min(image.shape[0], y + h + offset + 5),
                      max(0, x - offset): min(image.shape[1], x + w + offset + 5)]
        return imageCrop, image
    return None, image

# Resize the image to fit the required size
def resizeImage(imageCrop, imageSize, h, w):
    croppedImageWithWhiteBg = np.ones((imageSize, imageSize, 3), np.uint8) * 255
    aspectRatio = h / w
    if aspectRatio > 1:
        k = imageSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imageCrop, (wCal, imageSize))
        wGap = math.ceil((imageSize - wCal) / 2)
        croppedImageWithWhiteBg[:, wGap:wGap + wCal] = imgResize
    else:
        k = imageSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imageCrop, (imageSize, hCal))
        hGap = math.ceil((imageSize - hCal) / 2)
        croppedImageWithWhiteBg[hGap:hGap + hCal, :] = imgResize
    return croppedImageWithWhiteBg

# Save the image
def saveImage(croppedImageWithWhiteBg, folder, counter):
    try:
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", croppedImageWithWhiteBg)
        counter += 1
        print(f"Image number {counter} saved")
    except Exception as e:
        print(f"Error saving image: {e}")

# Main function
def main():
    cap, detector = initialiseCamera()
    offset = 30
    imageSize = 300
    counter = 0
    folder = 'savedDataModel5/notAugmented/Unknown'
    
    while True:
        imageCrop, image = captureHandImage(cap, detector, offset, imageSize)
        if imageCrop is not None:
            imgWhite = resizeImage(imageCrop, imageSize, *imageCrop.shape[:2])
            cv2.imshow("Image White", imgWhite)
        
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == ord("g"):
            saveImage(imgWhite, folder, counter)
            counter += 1
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()