import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imghdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def predict(img_cv):
    # Load the pre-trained model
    model = tf.keras.models.load_model((os.path.join('brain_tumor_model.h5')))

    # Resize the image to the model input size
    resize = tf.image.resize(img_cv, (256, 256))

    # Normalize the image (scale pixel values to range [0, 1])
    resized_img = resize / 255.0

    # Make prediction (expand dimensions to add batch size of 1)
    yhat = model.predict(np.expand_dims(resized_img, 0))

    # Output the prediction result
    if yhat[0][0] > 0.5:
        return 'Brain Tumor Detected'
    else:
        return 'No Brain Tumor Detected'