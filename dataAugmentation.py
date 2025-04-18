# dataAugmentation.py is a Python script that reads images from a folder, applies data augmentation techniques, and saves the augmented images to a new folder.

import os
import cv2
import numpy as np
import time
import math

# Define the path to the data that isn't augmented
dataDir = '/Users/shoubhitabhin/Documents/VSCode Projects/ALevel/ALevelNEA/JanMMLV3/savedDataModel5/notAugmented'

# Ensure augmented data folder exists
augmentedDataDir = os.path.join(dataDir, 'augmented')

if not os.path.exists(augmentedDataDir):
    os.makedirs(augmentedDataDir)

# Loop through each letter folder in the saved data
for letterFolder in os.listdir(dataDir):
    letterPath = os.path.join(dataDir, letterFolder)

    # Skip files like .DS_Store (ERROR ENCOUNTERED WHEN TESTING)
    if not os.path.isdir(letterPath):
        continue

    # Just for me
    print(f"Processing images for letter: {letterFolder}")
    
    for imageName in os.listdir(letterPath):
        imagePath = os.path.join(letterPath, imageName)

        # Skip non-image files
        if not imagePath.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Read the image
        image = cv2.imread(imagePath)
        if image is None:
            continue
        
        augmentedImages = []

        # Flipping the images
        flippedImage = cv2.flip(image, 1)
        augmentedImages.append(flippedImage)

        # Rotation
        rows, cols = image.shape[:2]
        angle = np.random.uniform(-15,15) # Means the image is rotated anywhere between 15 degrees and 15 degrees
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotatedImage = cv2.warpAffine(image, M, (cols, rows))
        augmentedImages.append(rotatedImage)

        zoomFactor = np.random.uniform(0.9, 1.1)
        zoomedImage = cv2.resize(image, None, fx=zoomFactor, fy=zoomFactor)
        zoomedImage = cv2.resize(image, None, fx=zoomFactor, fy=zoomFactor)
        h, w = zoomedImage.shape[:2]
        zoomedImageThatIsNowPadded = np.zeros((300, 300, 3), dtype=np.uint8)
        xOffset = (300 - w) // 2
        yOffset = (300 - h) // 2
        if xOffset >= 0 and yOffset >= 0:
            zoomedImageThatIsNowPadded[yOffset:yOffset + h, xOffset:xOffset + w] = zoomedImage
        else:
            # If the zoomed image is bigger than 300x300, resize it down since that is what the model is expecting
            zoomedImage = cv2.resize(zoomedImage, (300, 300))
        augmentedImages.append(zoomedImageThatIsNowPadded)

        # Applies the transformation matrix to the image using affine warping, which is a special type of transformation matrix
        horizontalTranslation = np.random.randint(-10, 10)
        verticalTranslation = np.random.randint(-10, 10)
        shiftingMatrix = np.float32([[1, 0, horizontalTranslation], [0, 1, verticalTranslation]])
        shiftedImage = cv2.warpAffine(image, shiftingMatrix, (cols, rows))
        augmentedImages.append(shiftedImage)

        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisyImage = cv2.add(image, noise)
        augmentedImages.append(noisyImage)

        """
        Possible reason for failure of B and D since they are rectangualr in shape, solution from ChatGPT below
        # Resize to a new size (e.g., 64x64)
        img_resize = cv2.resize(img, (64, 64))
        augmented_images.append(img_resize) 
        """

        #  --- The code below is from ChatGPT --- 

        # Resize while maintaining aspect ratio and padding to 300x300
        h, w = image.shape[:2]
        scale = 300 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h))

        # Create a blank black image (300x300) and center the resized image
        padded_img = np.zeros((300, 300, 3), dtype=np.uint8)
        xOffset = (300 - new_w) // 2
        yOffset = (300 - new_h) // 2
        padded_img[yOffset:yOffset + new_h, xOffset:xOffset + new_w] = resized_img

        augmentedImages.append(padded_img)

        # --- End of ChatGPT code ---

        # Save augmented images
        for idx, augmentedImage in enumerate(augmentedImages):
            augmentedImageName = f"{letterFolder}_{imageName.split('.')[0]}_aug_{idx+1}.jpg"
            augmentedImagePath = os.path.join(augmentedDataDir, letterFolder, augmentedImageName)
            if not os.path.exists(os.path.join(augmentedDataDir, letterFolder)):
                os.makedirs(os.path.join(augmentedDataDir, letterFolder))
            
            cv2.imwrite(augmentedImagePath, augmentedImage)

        print(f"Augmented {imageName} and saved {len(augmentedImages)} new images.")

print("Data augmentation completed!")