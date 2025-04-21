# SAVE KERAS MODEL AS A H5 MODEL FOR THE CONFUSION MATRIX

import tensorflow as tf

# Load the existing model (assuming it's in the .keras format)
model = tf.keras.models.load_model('/Users/shoubhitabhin/Documents/VSCode Projects/ALevel/ALevelNEA/JanMMLV3/CNNModels/modelWithUnknown2.keras')

# Save it in .h5 format
model.save('modelWithUnknown2.h5')

print("Model has been successfully saved in H5 format.")