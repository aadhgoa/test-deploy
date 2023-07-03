import io

# Import required libraries
from PIL import Image
import tensorflow as tf
import cv2
import segmentation_models as sm
from matplotlib import pyplot as plt
import numpy as np

# Define a function to load the pre-trained segmentation model
def get_segmentator():
    # Load the pre-trained segmentation model
    model = tf.keras.models.load_model("backend/model_50.h5",
                                        custom_objects={
                                            'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss,
                                            'iou_score':sm.metrics.iou_score
                                        })
    return model

# Define a function to get the segmented image
def get_segments(model, image):
    # Define the image dimensions
    width=256
    height=256
    
    # Resize the input image
    image = cv2.resize(image, (width, height))
    
    # Normalize the image pixel values
    x = image/255.0
    
    # Add an extra dimension to the image array
    x = np.expand_dims(x, axis=0)
    
    # Use the pre-trained segmentation model to predict the segmentation mask
    y_pred = model.predict(x, verbose=0)[0]
    
    # Return the predicted segmentation mask
    return y_pred
