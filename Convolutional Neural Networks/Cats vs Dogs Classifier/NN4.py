# My fourth neural network, (convolutional) this is a simple classification network model
# the data we are using is a cats and dgos dataset, this has pictures of cats and dogs
# and the network should be able to predict a cat or dog after being trained
# because our data is bigger im going to use my gpu (python -m pip install tensorflow-metal)
# if the data was smaller i would use my cpu (python -m pip uninstall tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cpu the cores are faster but can use less at once

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
we have a single hidden layer of 128 neurons, which is more than our output and less than double our input.
this layer is fully connected to the output, and the activation function is 'relu'

#--output--#
Our output is 1 neurons representing 0 or 1 for our classes. these will return values between
0-1, giving us percentages of what our network thinks the chance of it being this class is. using the
sigmoid activation function we will see which number it leans towards and classifying it

#--extra info--#
our network should improve our output to make it so that the actual class is at 1 and the rest are at 0. 
our optimiser is adam
our loss function is binary crossentropy (we only have two classes)

#------------------------#

'''

import pickle
import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#loading in our data from pickle
x = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

img_size = 80

#turning our data in np arrays
x = np.array(x).reshape(-1,img_size, img_size, 1)
y = np.array(y)

# normalising our data
x = x/255

# creating our test data
x_test = x[:1000]
y_test = y[:1000]
x = x[:len(x)-1000]
y = y[:len(y)-1000:]

'''
#creating a base model
model = keras.Sequential()

#creating our first layer
model.add(Conv2D(64, (3,3), input_shape=x.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#creating our second layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#creating our third layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))

#temporary 4th layer
model.add(Dense(64, activation='relu'))

# creating our output
model.add(Dense(1, activation='sigmoid'))

#compiling our model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs = 6, validation_split=0.1, batch_size=32)

model.save('dog-cat-2.model')
print("MODEL SAVED ")
'''
#loading our model
model = keras.models.load_model('dog-cat.model')

#creating a label
class_names = ['cat', 'dog']

#getting our loss and accuarcy
test_loss, test_acc = model.evaluate(x_test, y_test)

print('loss: ',test_loss, 'accuracy: ', test_acc)

#getting our predictions
pred = model.predict(x_test)

for x in range(len(x_test)):
    plt.imshow(x_test[x], cmap='gray')
    plt.title('Predicted: '+ str(class_names[round(pred[x][0])]))
    plt.xlabel('Actual: ' +str(class_names[y_test[x]]))
    plt.show()

'''
#-- results --#
by looking at our results we can see that the model does well, however when there is an image that is blurry
the model will look at just the shape and this can make it wrong, however this is rare.
'''