import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# loading in our json file
file = json.loads(open('intents.json').read())

# loading in our previous data
words = pickle.load(open('words.pickle', 'rb'))
classes = pickle.load(open('classes.pickle', 'rb'))
model = load_model('chatbot-model.model')

# function for cleaning up our sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

d = clean_up_sentence("hello how are you")
print(d)