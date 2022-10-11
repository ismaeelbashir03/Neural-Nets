# My third neural network, this is a simple  digit classification network model
# the data we are using is the digit mnist dataset, this has handdrawn pictures of digits
# and the network should be able to classify the digits after being trained
# because our data is small im going to use my cpu (python -m pip uninstall tensorflow-metal)
# if the data was bigger i would use my gpu (python -m pip install tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cpu the cores are faster but can use less at once

'''
#------architecture------#

#--input--#
our input is a 2d array of greyscale values, (which we have normalised)
we can flatten our input to make it one dimensional, this gives us an array of size 784 (28x28).
this means we will have 784 neurons as our input.
this layer is fully connected to the hidden layer

#--hidden layer 1--#
we have a single hidden layer of 128 neurons, which is more than our output and less than double or input.
this layer is fully connected to the output, and the activation function is 'relu'

#--hidden layer 2--#
we have a single hidden layer of 128 neurons, which is more than our output and less than double or input.
this layer is fully connected to the output, and the activation function is 'relu'

#--output--#
Our output is 10 neurons each representing our digit classes. these will return values between
0-1, giving us percentages of what our network thinks the chance of it being this class is. the highest
percent will be our prediction. the activation function is 'softmax'

#--info--#
our network should improve our output to make it so that the actual class is at 1 and the rest are at 0. 
our loss function is going to be sparse catagorical cross entropy
our optimiser is adam

#--extra info--#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
#------------------------#

'''

import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
import cv2

# load in our data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalise our data to be between 0 and 1
train_images = train_images/255
test_images = test_images/255

#model has already been created
'''

#designing our model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28))) # input
model.add(keras.layers.Dense(128, activation='relu')) # hidden layer 1
model.add(keras.layers.Dense(64, activation='relu')) # hidden layer 2
model.add(keras.layers.Dense(10, activation='softmax')) # output

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, validation_split=0.1, epochs=10)

model.save('digits_model.h5')
'''

# loading in our model
model = keras.models.load_model('digits_model.h5')

# getting the loss and accuarcy of our model
test_loss, test_acc = model.evaluate(test_images, test_labels)



for x in range(1,8):

    # making a real results list
    real = [4,1,9,5,7,9,1]

    # reading in our image with cv2, the last part is only getting the shape and leaving out the colour
    # (only getting first channel)
    img = cv2.imread(f"digits/digit{x}.png")[:,:,0]
    #invert the image to predict it as it origionally is white on black
    # we also put it in an array cause the input format is a 2d array
    img = np.invert(np.array([img]))

    #running the image in our model
    pred = model.predict(img)
    #pred = model.predict(test_images)

    # showing our image and the preidction
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.title('(my own data) Predicted: '+str(np.argmax(pred[0])))
    plt.xlabel('Actual: '+ str(real[x-1]))
    plt.show()

for x in range(1,8):

    #running the image in our model
    pred = model.predict(test_images)
    #pred = model.predict(test_images)

    # showing our image and the preidction
    plt.imshow(test_images[x], cmap=plt.cm.binary)
    plt.title('(test data) Predicted: '+str(np.argmax(pred[x])))
    plt.xlabel('Actual: '+ str(test_labels[x]))
    plt.show()

'''
#-- results --#

by looking at the predictions we can see that the model is quite accuarcte but 
due to my different ways of writing a 1, which assume is not familiar to the model
as it may have not been in the training it guessed it wrong. instead it predicted 2
since the 1 i drew resembled a squished 2 also my 9 had a weird shapewhich the model 
recognised as a 1. i then added in a 9 and 1 that is more familiar and the model
guessed correctly. i assume the mispredictions were due to the 'blocky' nature of my drawings
as when tested with the test data, it performed well.

'''