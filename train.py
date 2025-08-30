import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the directories where your training data is stored
train_data_dir = 'train_data'  # The directory where 'healthy' and 'diseased' folders are stored

# Set up image data generators for loading and augmenting the images
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to be between 0 and 1
    shear_range=0.2,             # Randomly shear images
    zoom_range=0.2,              # Randomly zoom images
    horizontal_flip=True         # Randomly flip images horizontally
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,              # Directory containing training data
    target_size=(224, 224),      # Resize images to 224x224
    batch_size=32,               # Number of images to process per batch
    class_mode='binary'          # Binary classification: healthy vs diseased
)

# Define a simple Convolutional Neural Network (CNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model with an optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=10)

# Save the model to a file called model.h5
model.save('model2.h5')
