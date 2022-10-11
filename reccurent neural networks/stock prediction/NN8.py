# My eighth neural network, (reccurent) this is a regression network model
# the data we are using is stock data from the last 60 days, and the network
# should be able to predict the next day based on this after being trained.
# because our data is bigger im going to use my gpu (python -m pip install tensorflow-metal)
# if the data was smaller i would use my cpu (python -m pip uninstall tensorflow-metal)
# this is because when working with gpu's they can use more at once, but are slower whereas
# on the cpu the cores are faster but can use less at once

'''
#------architecture------#

#--input--#
our input is sequenced stock prices over 60 days as arrays

#--hidden layer 1--#
this layer is STLM layer which is a type of RNN, which can keep previous data into account
for its predictions (uses previous context). it does this by using three gates (each with NN'S)
the first is the forget gate which runs a sigmoid on the data to forget 
(0 - forget 0.5 - half forget, 1 - fully keep), this is added the 'memory'. the second 
gate is the input gate, decides what is added to memory using sigmoid (1 - add, 0.5 - add half, 0 - dont add)
this sigmoid is multiplied by a tanh of the input (values to be added), this is then added
to the memory. the third gate chooses what to part of the memory to send to the future layer
this is done by a sigmoid NN multiplied by a tanh and the output is sent forward. the whole 
memory is sent as output tho (doesnt take the third gate into account). this has 50 neurons 
and the activaion is 'tanh' and the dropout is 0.2
( this stops overfitting and makes NN robust by randomly turning off defined precent of nuerons
so that each neuron isnt specified and can be used genrally.)

#--hidden layer 2--#
another lstm layer with a dropout of 0.2

#--hidden layer 3--#
another lstm layer with a dropout of 0.2

#--output--#
Our output is 1 neuron . this will return values between
0-1, giving us predictions of the closing date of stocks (its been scaled). 
we can inverse the scalar to get the actual price predicted

#--extra info--#
our optimiser is adam
our loss function is mean squared error

epoch = 25
batch size = 32
#------------------------#
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 60, 50)            10400     
                                                                 
 dropout (Dropout)           (None, 60, 50)            0         
                                                                 
 lstm_1 (LSTM)               (None, 60, 50)            20200     
                                                                 
 dropout_1 (Dropout)         (None, 60, 50)            0         
                                                                 
 lstm_2 (LSTM)               (None, 50)                20200     
                                                                 
 dropout_2 (Dropout)         (None, 50)                0         
                                                                 
 dense (Dense)               (None, 1)                 51        
                                                                 
=================================================================
Total params: 50,851
Trainable params: 50,851
Non-trainable params: 0

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# loading in our data from the web
company = 'AAPL'

#setting our start and end date for training 
start = dt.datetime(2013,1,1)
end = dt.datetime(2022,8,8)

#scraping the web at yahoo for this data
data = web.DataReader(company, 'yahoo', start, end)

# processing our data
print(data)
# scaling our data between 1 and 0
scalar = MinMaxScaler(feature_range=(0,1))
# reshaping our data to be 1d (-1, means any)
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1,1)) 

# setting days we will look at before predicting
training_days = 60

# getting x and y train values
x_train = []
y_train  =[]

# looping from training_days to the amount of data we have
for x in range(training_days, len(scaled_data)):
    # setting our training data to be 60 day increments
    x_train.append(scaled_data[x-training_days : x,0]) # ',0' takes out of array, so x_train is 2d
    y_train.append(scaled_data[x,0])

# setting our training data to be numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# reshaping our data to have an extra dimension
x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], 1))

# designing our model

model = Sequential()
# first layer, setting 
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# second layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# third layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1))

opt = Adam(learning_rate=0.0005)

# compiling our model
model.compile(optimizer='adam', loss='mean_squared_error')

# training our model
model.fit(x_train, y_train, batch_size=32, epochs=25)

# getting our test data

#defining our start and end times
start_test = dt.datetime(2021,1,1)
end_test = dt.datetime.now()

# scraping for our test data
test_data = web.DataReader(company, 'yahoo', start_test, end_test)

print(test_data)
# getting our actual prices
actual_prices = test_data['Close'].values

# joining our test and training closing prices
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# defining the input for our model by making it the test data minus the training days
model_inputs = total_dataset[len(total_dataset) - len(test_data) - training_days:].values

# making our input 2d
model_inputs = model_inputs.reshape(-1, 1)

# scaling our data between 1 and 0
model_inputs = scalar.transform(model_inputs)

# making predictions on test data
x_test = []

# defining our test inputs
for x in range(training_days, len(model_inputs)):
    # starting from 0 and getting 60 (training days) closing prices each time
    x_test.append(model_inputs[x-training_days:x, 0])

# making our data an numpy array and reshaping it to add a dimension (3d)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# getting our predictions
pred_prices = model.predict(x_test)

# reversing our scaling so we can compare to actual data
pred_prices = scalar.inverse_transform(pred_prices)

# plotting the actual prices
plt.plot(actual_prices, color='black', label=f'Actual Price of {company}')

#plotting our predictions
plt.plot(pred_prices, color='green', label=f'Predicted Price of {company}')

# adding a title to the graph
plt.title(f"{company} Share Price")

# setting our x and y labels
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")

# showing our legend
plt.legend()

# showing our graph
plt.show()

# next day prediction

# getting the data 60 days before today
real_data = [model_inputs[len(model_inputs)+1-training_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# getting a prediction
pred = model.predict(real_data)
pred = scalar.inverse_transform(pred)

print("Predicted for CLOSING:", pred)

model.summary()

"""

as we can see from our results, just because the line follows the same pattern does not mean it is good
since our model predicts these patterns after it has already happened. our model predicts
a closing date for today (10/06/22) as 661.9203 for TSLA. 144.908 for AAPL. 2258.025 for GOOG.
i will update if it got close

"""