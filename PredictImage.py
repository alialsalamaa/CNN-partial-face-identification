import tensorflow as tf
import numpy as np
import cv2
from dictionary import folder_list

# define the path to the directory containing the custom model
model_path = r'.\models\PartialFaceModel.h5'

# load the model
model = tf.keras.models.load_model(model_path, compile = False)
model.compile()

# define the path to the image you want to predict
image_path = r'.\testres\MohG\MohG_01724_m_32_i_fr_nc_no_2016_2_e0_Ps_m.jpg'


# load the image and preprocess it
image = cv2.imread(image_path)
image = cv2.resize(image, (160, 160))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

# use the model to make predictions on the image
predictions = model.predict(image)

# load the list of class names
class_names = folder_list

# get the indices of the top 5 predicted probabilities
top5_indices = np.argsort(predictions[0])[::-1][:5]

# print the top 5 predicted class names and their probabilities
for i in top5_indices:
    class_name = class_names[i]
    probability = predictions[0][i]
    print(f"{class_name}: {probability:.2%}")