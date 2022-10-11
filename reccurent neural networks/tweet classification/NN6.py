# My sixth neural network, (reccurent) this is a simple classification network model
# the data we are using is a tweets dataset, this has tweet text
# and the network should be able to predict if the tweet is talking about a disaster
# after being trained.
# because our data is bigger im going to use my gpu (python -m pip install tensorflow-metal)
# if the data was smaller i would use my cpu (python -m pip uninstall tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cpu the cores are faster but can use less at once

'''
#------architecture------#

#--input--#
our input is sequenced words as arrays which have been tokenized

#--hidden layer 1--#
this layer is an Embedded layer, this is most always used when dealing with text input
as it clusters together similiar words, it does this by turning every word into a vector
and calculating the angle between each vector to get the word similarity. we have used 32
dimensions for our vectors, and our model will train to move these vectors together or
apart depending on their similarity. then we added a spatial dropout of 50 percent

#--hidden layer 2--#
this layer is STLM layer which is a type of RNN, which can keep previous data into account
for its predictions (uses previous context). it does this by using three gates (each with NN'S)
the first is the forget gate which runs a sigmoid on the data to forget 
(0 - forget 0.5 - half forget, 1 - fully keep), this is added the 'memory'. the second 
gate is the input gate, decides what is added to memory using sigmoid (1 - add, 0.5 - add half, 0 - dont add)
this sigmoid is multiplied by a tanh of the input (values to be added), this is then added
to the memory. the third gate chooses what to part of the memory to send to the future layer
this is done by a sigmoid NN multiplied by a tanh and the output is sent forward. the whole 
memory is sent as output tho (doesnt take the third gate into account). this has 64 neurons 
and the activaion is 'tanh' and the dropout is 0.5
( this stops overfitting and makes NN robust by randomly turning off defined precent of nuerons
so that each neuron isnt specified and can be used genrally.)


#--output--#
Our output is 1 neurons representing 0 or 1 for our classes. these will return values between
0-1, giving us percentages of what our network thinks the chance of it being this class is. using the
sigmoid activation function we will see which number it leans towards and classifying it

#--extra info--#
our network should improve our output to make it so that the actual class is at 1 and the rest are at 0. 
our optimiser is adam
our loss function is binary crossentropy (we only have two classes)

epoch = 15
batch size = 4
learning rate = 0.0005
#------------------------#

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 24, 4)             91180     
_________________________________________________________________
spatial_dropout1d (SpatialDr (None, 24, 4)             0         
_________________________________________________________________
lstm (LSTM)                  (None, 4)                 144       
_________________________________________________________________
dropout (Dropout)            (None, 4)                 0         
_________________________________________________________________
dense (Dense)                (None, 1)                 5         
=================================================================
Total params: 91,329
Trainable params: 91,329
Non-trainable params: 0

'''
# data handling imports
from tokenize import Token
import pandas as pd
import numpy as np

# neural net imports
import tensorflow as tf
import tensorflow.python.keras as keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, GlobalAveragePooling1D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# pre-proccessing imports
import re
import string
import nltk
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# reading in our data
test_data = pd.read_csv('data/test.csv')
view_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')


# preprocessing the data to remove urls and punction as they give no information
# to the nueral net

# getting a pattern for a url and removing it from the string
def remove_url(string):
    url = re.compile(r"https?://\S+|www\.\S")
    return url.sub(r"", string)

# removeing the punctuation
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

#using our preproccessing functions
test_data["text"] = test_data.text.map(remove_url)
test_data["text"] = test_data.text.map(remove_punct)

train_data["text"] = train_data.text.map(remove_url)
train_data["text"] = train_data.text.map(remove_punct)

#removing our stopwords (and, the, a, an, in), they will not give info for the classification

# downloading the stop words
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

# function to remove stopwords
def remove_stopwords(string):

    # gets every word (lower case) and checks if its in the stop words, if not add to list
    filtered_words = [word.lower() for word in string.split() if word not in stop_words]
    return " ".join(filtered_words)

# using our function
test_data["text"] = test_data.text.map(remove_stopwords)

train_data["text"] = train_data.text.map(remove_stopwords)

# now that our text has been filtered to text that we need, we need to now make it 
# into a usable input

# getting words from each data and making a counter for each word to get the unique amount
def unique_words(test_words, train_words):
    count = Counter()
    
    for text in test_words.values:
        for word in text.split():
            count[word] += 1

    for text in train_words.values:
        for word in text.split():
            count[word] += 1

    return count

# using our function to get a dict
word_dict = unique_words(test_data.text, train_data.text)

num_unique_words = len(word_dict)

#splitting our data (we can evaluate y_tes for our selves)
x_train = train_data.text.to_numpy()
y_train = train_data.target.to_numpy()

x_test = test_data.text.to_numpy()

# tokenizing our data (turning into a vector with n dimensions i.e. ax +by +cz = d)
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(x_train) # only on train data

# creating a word index for each word in our training data
word_index = tokenizer.word_index

# we can now create a sequence using tokenizer 
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

# here we find the max number of sequences in each training data to put for padding
max = 0

def find_max(max):
    for lst in x_train_seq:
        if len(lst) > max:
            max = len(lst)

    return max

# using the function
max = find_max(max)

# padding our data to be the same size
x_train_pad = pad_sequences(x_train_seq, maxlen=max, padding='post', truncating='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=max, padding='post', truncating='post')

# creating a reverse word index for decoding
rev_word_index = dict((index, word) for word, index in word_index.items())

# function to decode a sequence
def decode(sequence):
    return " ".join([rev_word_index.get(index, "?") for index in sequence])

"""
# now that all of the preparing is done, we can now design our model
model = Sequential()
# embedded layer where we define the number of words, vector dimensions, and input length
model.add(Embedding(num_unique_words, 4, input_length=max))
model.add(SpatialDropout1D(0.5))

# LSTM layer where nuerons is 2 and droput is 50 percent
model.add(LSTM(units=2))
model.add(Dropout(0.5))

# output layer
model.add(Dense(1, activation='sigmoid'))

# defining an optimizer
opt = Adam(learning_rate=0.0005)

# compiling our model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# training our model
model.fit(x_train_pad, y_train, epochs=15, batch_size=4, validation_split=0.2)

model.save('tweet-model5.model')
"""
# loading in our model (0.79, had an error in the tokenizer where i inputted test for only train)
model = keras.models.load_model('0.78-model.model')

#print(model.summary()) --- the trip was very good, pakistan was nice.

# getting predictions for our test data
predictions = model.predict(x_test_pad)

# rounding our predictions to 1 or 0
predictions = [1 if p > 0.5 else 0 for p in predictions]

# creating readable labels
labels = ["TWEET DOES NOT TALK ABOUT DISASTER", "TWEET TALKS ABOUT DISASTER"]

choose = ""

# creating input for choice
while (choose != "a" and choose != "b"):
    choose = input("Do you want to enter a tweet [enter 'A'], or go through test tweets [Enter 'B']: ")
    choose = choose.lower()

if choose == "b":
    #printing our predictions back to us
    for x,p in enumerate(predictions):

        #printing results
        print("Tweet:", view_data.text[x])
        print("predicted:", labels[p])
        
        # allowing us to stop looping through the predictions
        cont = ''

        while cont == '':
            cont = input("Continue [y/n]: ")
        
        if cont == 'N' or cont == 'n':
            break

elif choose == "a":
    done = False
    while done == False:
        #getting tweet and preprocessing it
        tweet = input("Enter a tweet about a distaster or not: ")

        tweet = remove_url(tweet)
        tweet = remove_punct(tweet)
        tweet = remove_stopwords(tweet)
        tweet = np.array([tweet])
        tweet = tokenizer.texts_to_sequences(tweet)
        tweet = pad_sequences(tweet, maxlen=max, padding='post', truncating='post')

        prediction = model.predict(tweet)

        if prediction[0] >= 0.5:
            pred = 1
        elif prediction[0] < 0.5:
            pred = 0

        print("Model predicted:", labels[pred])

        d = input("keep going? [y/n]: ")

        if d.lower() == "n":
            done = True
        elif d.lower() == "y":
            done = False


"""
by looking at our results we can see our model was a success it almost always predicts
right. at first i tried using 64 units in the lstm and only using 1, and my embedded 
had 128 dimensions. but i ran into overfitting as my val_loss was increasing and my 
val_acc was decreasing every epoch and my acc and loss were doing the opposite.
since my val_loss was drastically higher than my loss i knew it was overfitting and
decided to cut the units in my lstm to 4 and my embedded to 4 through trial and error.
then by running this config with a batch size of 4 and 15 epochs and a learning rate of 
0.0005, i ended up with a 79.65 percent accuracy (roughly 80 percent), whihc is great.

"""

    
