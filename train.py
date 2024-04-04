import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from PIL import UnidentifiedImageError
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from data_preprocess import X_test,X_train,test_datagen,train_datagen,y_test,y_train,load_images_from_folders,recyclable_folder,non_recyclable_folder
from model_build import model


# Model Training
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // 32,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=len(X_test) // 32)

