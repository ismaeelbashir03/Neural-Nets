# My second neural network, this is a text classification network model
# the data we are using is the imdb review dataset, this has words set encoded as numbers
# and the network should be able to predict the review type (neg or pos)
# because our data is small im going to use my cpu (python -m pip uninstall tensorflow-metal)
# if the data was bigger i would use my gpu (python -m pip install tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cpu the cores are faster but can use less at once

'''
#------architecture------#

#--input--#
our input is an array of number values, (which we have a dictionary for)
we have padded all the reviews to 250 words, this gives us an array of size 250.

#--embedded layer--#
this layer is used for grouping together words based on positivity or negativity. this is done by using 10,000
word vectors with 16 dimensions. the model calulates the angle between word vectors to check to see its similarity
(e.g. good and great have an angle of 2, whereas good and bad have an 180) the model moves these vectors together 
or apart by looking at the words surrounding both words words.

#--gloabl average pooling --#
this layer is used to scale down the output of the previous layer from 16 dimensions to one by averageing them out

#--hidden layer--#
we have a single hidden layer of 16 neurons, which is more than our output and less than double or input.
this layer is fully connected to the output, and the activation function is 'relu'. this is where the model will classify
the data

#--output--#
Our output is 1 neuron representing our review type. this will return a value of
0 or 1, giving us 1 for a good review and 0 for a bad review. the activation function is 'sigmoid'

#--extra info--#

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________

#------------------------#
'''

import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np

# getting our data from keras datasets
data = tf.keras.datasets.imdb

# splittign our data
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)

#getting the word dictionary ({1: 'hello'})
word_index = data.get_word_index()

#moving up the values of the index by 3 (they start at 1) and adding in 4 'special' words
word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PADDING>'] = 0 # to pad out the reviews
word_index['<START>'] = 1 # to show the start of the review
word_index['<UNK>'] = 2 # for unknown words
word_index['<UNUSED>'] = 3 # for unused words

# reversing our index to point to a word instead of key
reverse_word_index = dict([(v, k) for k, v in word_index.items()])

#padding out all our reviews to 250 words for input
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PADDING>"], padding = 'post', maxlen = 250)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PADDING>"], padding = 'post', maxlen = 250)

# returns the words for each number word in a review passed in so it is readable
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#already created model
'''
# creating the model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# compliling our model, we use the binary_crossentropy loss function as we have only two outputs 0 and 1
# this function will specifically calc the loss for the prediction compared to the real answer of 0 or 1
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# splitting our training data to get validation data
# this is done to get an accuracy  if the model on new 
# data while the model is training we do this to see how 
# well our model is doing
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# training our model, we set our batch size to 512 which means the number of reviews our model will train 
# with at once. verbose is how detailed the output is (defualt is 1).
model_fit = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

model.save('model.h5')
'''

model = keras.models.load_model('model.h5')

def encode_review(text):
    encoded = [1]

    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

review = ['bad', 'good']

with open('review.txt', encoding='utf-8') as file:
    for line in file.readlines():
        nline = line.replace(',', "").replace('.', "").replace('(', "").replace(')', "").replace(':', "").replace('\"', "").strip().split(" ")
        encode = encode_review(nline)
        encode = tf.keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PADDING>"], padding = 'post', maxlen = 250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(review[round(predict[0][0])])

'''
#getting the accuracy and loss from our model
results = model.evaluate(test_data, test_labels)

# printing a test prediction
review_index = 1
test_review = test_data[review_index]
predict = model.predict([test_review])
print('Review: ')
print(decode_review(test_review))
print('Prediction: ', review[round(predict[0][0])])
print('Actual ', review[test_labels[review_index]])
print('Model loss:',results[0],'Model accuracy:', results[1])

'''
