# run with this script always -- python3 -m modelCreationAndTraining.train

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from modelCreationAndTraining.model import buildModel
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class signLanguageTrainer:
    def __init__(self, data_dir, batch_size=32, epochs=5):  
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs 

    def train(self):
        # Only rescales data (no augmentation since this script uses the augmented data )
        train_datagen = ImageDataGenerator(rescale=1./255)

        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(300, 300),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'evaluation'),
            target_size=(300, 300),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        # Print class indices to verify correct encoding (for personal testing only)
        print("Class indices:", train_generator.class_indices)

        # Compute class weights to handle potential class imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Build and compile the model
        num_classes = len(train_generator.class_indices)
        model = buildModel(inputShape=(300, 300, 3), numClasses=num_classes)

        # Early stopping based on validation accuracy
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // self.batch_size),  # Prevent steps = 0
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=max(1, validation_generator.samples // self.batch_size),  # Prevent steps = 0
            callbacks=[early_stopping],
            class_weight=class_weight_dict
        )

        # Save the trained model
        model.save('CHANGETHENAMEOFTHIS.keras') # ADD MODEL NAME HERE MAKE SURE IT IS DIFFERENT FROM PREVIOUS MODEL NAMES OR CRASH BANG POW!
        print("Model training complete and saved.")

if __name__ == "__main__":
    data_dir = '/Users/shoubhitabhin/Documents/VSCode Projects/ALevel/ALevelNEA/JanMMLV3/savedDataModelWithUnknown2/trainOnThese'
    trainer = signLanguageTrainer(data_dir)
    trainer.train()