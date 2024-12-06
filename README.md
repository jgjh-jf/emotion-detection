# emotion-detection
used python and ML to detect emotion by training many images

# this project was made and tested in google collab

import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cnn=Sequential()
cnn.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(16,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())

cnn.add(Dense(64,activation='relu'))
cnn.add(Dense(32,activation='relu'))
cnn.add(Dense(16,activation='relu'))
cnn.add(Dense(8,activation='relu'))
cnn.add(Dense(4,activation='relu'))
cnn.add(Dense(1,activation='sigmoid'))

cnn.compile(loss='binary_crossentropy', optimizer="adam")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "/content/drive/Othercomputers/My Computer/train-",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        "/content/drive/Othercomputers/My Computer/data",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cnn.fit(
        train_generator,
        steps_per_epoch=20,
        epochs=30,
        validation_data=test_generator,
        )

from keras.preprocessing import image
img = image.load_img("/content/african-american-depressive-sad-broken-260nw-475790911.jpg",target_size=(64,64))
img = image.img_to_array(img)
import numpy as np

img # will give output of image(input by user above) of array

img = np.expand_dims(img,axis=0)
p = cnn.predict(img)
if p[0][0]==1:
  print("Depressed")
else:
  print("Not Depressed")

  #will give output as if image given above is depressed or not
  
        
