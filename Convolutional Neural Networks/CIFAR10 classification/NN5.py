# My fifth neural network, (convolutional) this is a simple classification network model
# the data we are using is the CIFAR10 dataset, this has pictures of different things
# and the network should be able to predict each class after being trained
# because our data is small im going to use my cpu (python -m pip uninstall tensorflow-metal)
# if the data was bigger i would use my gpu (python -m pip install tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cou the cores are faster but can use less at once

'''
#------architecture------#

#--input--#
our input is a 2d array of greyscale values, (which we have normalised)

#--hidden layer 1--#
this layer is convold2d layer, this filters the images by getting the dot product of the
filter matrix with each specified size of an image. a convolution is made of all these into
a matrix, this matrix is then pooled to a 2x2 matrix by using the max poolling layer
our filters are 3x3 and there are going to be 64, activation = relu

#--hidden layer 2--#
this layer is convold2d layer, this filters the images by getting the dot product of the
filter matrix with each specified size of an image. a convolution is made of all these into
a matrix, this matrix is then pooled to a 2x2 matrix by using the max poolling layer
our filters are 3x3 and there are going to be 64, activation = relu

#--hidden layer 3--#
we have a single hidden layer of 128 neurons, which is more than our output and less than double our input.
this layer is fully connected to the output, and the activation function is 'relu'

#--hidden layer 4--#
we have a single hidden layer of 64 neurons, which is more than our output and less than double our input.
this layer is fully connected to the output, and the activation function is 'relu'

#--output--#
Our output is 10 neurons representing 0 or 1 for our classes. these will return values between
0-1, giving us percentages of what our network thinks the chance of it being this class is. using the
softmax activation function we will see which number it leans towards and classifying it

#--extra info--#
our network should improve our output to make it so that the actual class is at 1 and the rest are at 0. 
our optimiser is adam
our loss function is binary crossentropy (we only have two classes)

#------------------------#
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import datasets, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow.python.keras as keras

# getting our data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#normalisign our data
x_train = x_train/255
x_test = x_test/255

# creating labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
#designing our model
model = Sequential()

#layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#layer2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#layer4
model.add(Flatten())
model.add(Dense(256, activation='relu'))

#layer5
model.add(Dense(128, activation='relu'))

#output
model.add(Dense(10, activation='softmax'))

#compiling our model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=20)

model.save('cifar10.model')
print('model saved')

'''
model = keras.models.load_model('cifar10.model')

pred = model.predict(x_test)

for x in range(len(x_test)):
    plt.imshow(x_test[x])
    plt.title('Predicted: ' + str(classes[np.argmax(pred[x])]))
    plt.xlabel('Actual: '+ str(classes[y_test[x][0]]))
    plt.show()