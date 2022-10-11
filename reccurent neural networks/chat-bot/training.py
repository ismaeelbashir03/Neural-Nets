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



#------------------------#
'''
import random
import json
import pickle
from tabnanny import verbose
import numpy as np

import nltk
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, SpatialDropout1D
from keras.optimizers import Adam

lemmatizer = WordNetLemmatizer()

# loading in our json file
file = json.loads(open('intents.json').read())

# initialising our lists
words = []
documents = []
ignore_letters = ['?', "!", ".", ","]

# going through each intent and storing the words in words[] and 
# tags in classes[] and each word to a tag in the documents[]
for intent in file['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

# lemmatizing our words (changing them to their base word)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# using set to get rid of duplicates and using sorted to turn it back into a list
words = sorted(set(words))

# saving our words and classes
pickle.dump(words, open('words.pickle', 'wb'))

# shuffling our data
random.shuffle(documents)

# creating x anf y values
x_train = []
y_train = []

for text, tag in documents:
    x_train.append(text)
    y_train.append(tag)

# getting num of unique words
num_unique_words = len(words)

# tokenizing our data (turning into a vector with n dimensions i.e. ax +by +cz = d)
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(x_train) # only on train data

# creating a word index for each word in our training data
word_index = tokenizer.word_index

# we can now create a sequence using tokenizer 
x_train_seq = tokenizer.texts_to_sequences(x_train)

# here we find the max number of sequences in each training data to put for padding
max = 0

def find_max(max):
    for lst in x_train:
        if len(lst) > max:
            max = len(lst)

    return max

# using the function
max = find_max(max)

# padding our data to be the same size
x_train_pad = pad_sequences(x_train_seq, maxlen=max, padding='post', truncating='post')

for x,y in enumerate(y_train):
    if y == "greetings":
        y_train[x] = 0
    elif y == "goodbye":
        y_train[x] = 1
    elif y == "age":
        y_train[x] = 2
    elif y == "name":
        y_train[x] = 3
    elif y == "features":
        y_train[x] = 4
    elif y == "stocks":
        y_train[x] = 5

# creating a reverse word index for decoding
rev_word_index = dict((index, word) for word, index in word_index.items())
            
# function to decode a sequence
def decode(sequence):
    return " ".join([rev_word_index.get(index, "?") for index in sequence])

# make our data into numpy arrays
x_train_pad = np.array(x_train_pad)
y_train = np.array(y_train)


# now that our data is prepared we can build our model
model = Sequential()
#an embedded layer to group together words
model.add(Embedding(num_unique_words, 64, input_length=max))
model.add(SpatialDropout1D(0.5))

#adding our lstm layer
model.add(LSTM(units=128, recurrent_dropout=0.5))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Dropout(0.5))

#output layer
model.add(Dense(6, activation='softmax'))


# defining our learning rate in our optimiser
opt = Adam(learning_rate=0.0005)

# compiling our model
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training our model
model.fit(x_train_pad, y_train, epochs=200, batch_size=5)

# saving our model
model.save('chatbot-model.model')


# loading our model
model = load_model('chatbot-model.model')

#--- UI ---#

#readable labels
classes = ["greetings", "goodbye", "age", "name", "features", "stocks"]

# clearing our console
import os
clear = lambda: os.system('clear')
clear()

done = False

while done == False:
    say = input("you: ")
    say = [lemmatizer.lemmatize(word) for word in say.split()]
    say = [say]
    say = tokenizer.texts_to_sequences(say)
    say = x_train_pad = pad_sequences(say, maxlen=max, padding='post', truncating='post')
    say = np.array(say)

    pred = model.predict(say, verbose=0)
    pred = np.argmax(pred)

    for intent in file['intents']:
        if intent['tag'] == classes[pred]:
            rand_index = random.randint(0, len(intent['responses'])-1)
            print("Neural:", intent['responses'][rand_index])

    
"""
by testing with our chatbot i can say that this type of model relies heavily on training data, 
since we have made little it can be a bit weird sometimes, but mostly it is correct.
when i use a '?' at the end of a word it doesnt seem to recognise it, if i were to
go back i would fix this.


"""