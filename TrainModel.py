from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import LeakyReLU
import numpy as np
from tensorflow import keras
import tensorflow as tf
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from architecture import * 
import os

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

train_dir = r'.\trainres'
test_dir = r'.\testres'
val_dir = r'.\valres'


# Automatically detect the number of classes in each directory
num_classes = len(os.listdir(train_dir))
folders = num_classes

# Define image size
img_size = (160, 160)

# Create an image data generator with data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Create an image data generator for the test and validation sets (no data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset using the image data generators
batch_size = 512
train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                               batch_size=batch_size, class_mode='sparse')
test_data = test_datagen.flow_from_directory(test_dir, target_size=img_size,
                                             batch_size=batch_size, class_mode='sparse')
val_data = val_datagen.flow_from_directory(val_dir, target_size=img_size,
                                            batch_size=batch_size, class_mode='sparse')





model = FaceResNet()
# model.summary()
print('Loaded Model')
model.load_weights("facenet_keras_weights.h5")
for layer in model.layers:
 layer.trainable = False
inputs = keras.Input(shape=img_size + (3,))
x = inputs

mean = np.array([127.5] * 3)
var = mean ** 2

# removing last layers and adding new layers 
model = Model(inputs=model.inputs, outputs=model.layers[-3].output)

x = model(x, training=False)
x = Dense(1024)(x)
x= LeakyReLU(alpha=0.03)(x)
x = Dense(512)(x)
x= LeakyReLU(alpha=0.03)(x)
x = Dense(128, activation='relu')(x)
x= BatchNormalization()(x)
outputs = Dense(folders, activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', 
metrics=['accuracy'])










# Train the model on the training data and validate on the validation data
epochs = 50
history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[reduce_lr]) #added callback

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)


# Plot the training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)

# Save the trained model to a chosen directory
save_dir = r'C:\Users\openu\OneDrive\Desktop\Master\models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save(os.path.join(save_dir, 'pfrproto5.h5'))