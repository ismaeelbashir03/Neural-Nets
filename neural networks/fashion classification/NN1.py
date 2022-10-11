# My first neural network, this is a simple classification network model
# the data we are using is the fashion mnist dataset, this has pictures of clothes
# and the network should be able to predict the type of clothing after being trained
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

#--hidden layer--#
we have a single hidden layer of 128 neurons, which is more than our output and less than double or input.
this layer is fully connected to the output, and the activation function is 'relu'

#--output--#
Our output is 10 neurons each representing our clothes classes. these will return values between
0-1, giving us percentages of what our network thinks the chance of it being this class is. the highest
percent will be our prediction. the activation function is 'softmax'

#--extra info--#
our network should improve our output to make it so that the actual class is at 1 and the rest are at 0. 

#--image--#

input   hidden-layer    output
0           0               0
0           0               0
0           0               0
0           0               0
0           0               0

#------------------------#
'''
import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import matplotlib.pyplot as plt

#our data is read in by keras' datasets
data = tf.keras.datasets.fashion_mnist

# splitting our data into training and testing
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# normalising our data to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

# creating readable class names for our output
class_names = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "bag", "Ankle Boot"]

# creating our model 
model = keras.Sequential()
# adding our input layer (whihc flattens the data)
model.add(keras.layers.Flatten(input_shape = (28,28)))
# adding our fully connected hidden layer
model.add(keras.layers.Dense(128, activation = 'relu'))
# adding our fully connected output layer
model.add(keras.layers.Dense(10, activation = 'softmax'))

#compliling our model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])

#training our model
model.fit(train_images, train_labels, epochs = 8)

#getting our loss and accuarcy
test_loss, test_acc = model.evaluate(test_images, test_labels)

#printing this back to us
print('Tested accuracy: ',test_acc)

#passing in an image of an ankle boot (close the image to continue the program)
predictions = model.predict(test_images)

# showing the first 5 predictions
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap= plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title('Prediction: ' + class_names[np.argmax(predictions[i])])
    plt.show()