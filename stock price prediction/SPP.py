#https://www.youtube.com/watch?v=QIUxPv5PJOY&ab_channel=ComputerScience
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
#LSTM stands for Long-Short Term Memory
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('PLUG', data_source='yahoo', start='2017-03-03', end='2022-02-27')
#show the data
print(df)

#Get the number of rows and columns in the dataset
#print(df.shape)
 #shape is (2003, 6) 2003 rows, 6 columns
 
 #visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()

#create a new dataframe with only the closing price column
data = df.filter(['Close'])
#convert the dataframe to a numpy array
dataset = data.values
#print(dataset)
#Get the number of rows to train the model on (80% of the dataset)
training_data_len = math.ceil(len(dataset) * .8)
#print(training_data_len)
 #1603

#Scale the data (aka preprocess the data)
#MinMaxScaler transforms features by scaling each feature to a given range
scaler = MinMaxScaler(feature_range=(0,1))
#dataset would be scaled to between 0 and 1
scaled_data = scaler.fit_transform(dataset)
#print(scaled_data)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :] #, :] is not necessary here, because there is only one column
#print(train_data)
#Split the data into x_train and y_train data sets
#x_train is the training features/independent variable
x_train = []
#y_train is the target variable/dependent variable
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #would have 60 values tuple in an array for each run
    y_train.append(train_data[i,0]) #would have 1603-60=1543 values after the loop finished
    
    # if i<=61:
    #     print("x_train: ", x_train)
    #     print("y_train: ", y_train)
    #     print()
        
#print(x_train)
#print(len(y_train))

#Convert the x_train and y_train to numpy arrays, we use them to train the LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data, LSTM expect the input to be 3D (no. of features, no. of timestamp, no. of sample)
#1543 is the row, 60 is the 60 values in tuples, and 1 is the closing price
# print(x_train.shape)
# x_train = np.reshape(x_train, (1543,60,1))
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
# print(x_train.shape)

#Build the LSTM model
model = Sequential()
#50 is the neuron, return sequence is true because it gonna feed into another LSTM layer,  specifies the data size thats gonna feed into the layer
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
#Add a dense layer to the model, 25 is the neuron
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
#optimizer is use to improve upon the loss function, loss function is used to measure how well the model did on training
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
#epochs is the number of iteration when the entire dataset is pass forward or backward through a neural network
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing data set
#Create a new array containing scaled value from index 1542 to 2003
test_data = scaled_data[training_data_len-60: , :]
#Create the data sets x_test and y_test
x_test = []
#y_test would be all of the value that we want our model to predict (actual test value)
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    #x_test contains past 60 values
    x_test.append(test_data[i-60:i, 0])
    
#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#Get the models predicted price value
predictions = model.predict(x_test)
#inverse_transform is kinda unscaling the value
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
# evaluate the performance of our model
# lower RMSE = better fit
rmse =np.sqrt(np.mean(((predictions- y_test)**2)))
rmse = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
rmse = np.sqrt(((predictions - y_test) ** 2).mean())
# print(rmse)
    
#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

#Show the valid and predicted prices
# print(valid)

#Get the quote
apple_quote = web.DataReader('PLUG', data_source='yahoo', start='2012-01-01', end='2022-02-28')

#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days 
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# #Get the quote
# apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2012-12-18', end='2019-12-18')
# print(apple_quote2['Close'])