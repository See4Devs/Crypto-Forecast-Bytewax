import calendar
import datetime
import csv
import re
from bytewax import run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

def file_input(market,FtxClient, unixStartDate, unixEndDate):
    resultsBTC = FtxClient.get_historical_prices(market, 3600, unixStartDate, unixEndDate)
    for lineBTC in resultsBTC:
        yield 1, lineBTC
        
def getOpenValue(line):
    return line['open']

#Run dataflow and write to CSV file
def runDataFlow(filename, market, flow, FtxClient,  unixStartDate, unixEndDate):
    with open(filename+'.csv', 'w') as f:
        header=['Open']
        writer = csv.writer(f)
        writer.writerow(header)
        #Run the Bytewax dataflow
        for epoch, item in run(flow, file_input(market, FtxClient,  unixStartDate, unixEndDate)):
           #print(item)
            writer.writerow([item])

def create_dataset(dataframe):
    xArray = []
    yArray = []
    for i in range(50, dataframe.shape[0]):
        xArray.append(dataframe[i-50:i, 0])
        yArray.append(dataframe[i, 0])
    xArray = np.array(xArray)
    yArray = np.array(yArray)
    return xArray,yArray

def cryptoForcast(Filename, market, Bytewaxflow, FtxClient,  unixStartDate, unixEndDate):
    runDataFlow(Filename, market, Bytewaxflow, FtxClient,  unixStartDate, unixEndDate)
    df = pd.read_csv(Filename+'.csv')
    df.shape
    df = df['Open'].values
    df = df.reshape(-1, 1)
    df.shape
    datasetTrain = np.array(df[:int(df.shape[0]*0.8)])
    datasetTest = np.array(df[int(df.shape[0]*0.8):])
    scaler = MinMaxScaler(feature_range=(0,1))
    datasetTrain = scaler.fit_transform(datasetTrain)
    datasetTest = scaler.transform(datasetTest)

    xtrain, ytrain = create_dataset(datasetTrain)
    xtest, ytest = create_dataset(datasetTest)

    modelCrypto = Sequential()
    modelCrypto.add(LSTM(units=96, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    modelCrypto.add(Dropout(0.2))
    modelCrypto.add(LSTM(units=96, return_sequences=True))
    modelCrypto.add(Dropout(0.2))
    modelCrypto.add(LSTM(units=96, return_sequences=True))
    modelCrypto.add(Dropout(0.2))
    modelCrypto.add(LSTM(units=96))
    modelCrypto.add(Dropout(0.2))
    modelCrypto.add(Dense(units=1))

    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))\

    modelCrypto.compile(loss='mean_squared_error', optimizer='adam')
    modelCrypto.fit(xtrain, ytrain, epochs=50, batch_size=32)
    modelCrypto.save(Filename+'prediction.h5')
    modelCrypto = load_model(Filename+'prediction.h5')

    ## Visualization results
    predictions = modelCrypto.predict(xtest)
    predictions = scaler.inverse_transform(predictions)
    ytest_scaled = scaler.inverse_transform(ytest.reshape(-1, 1))
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(ytest_scaled, color='red', label=Filename+' Original price')
    plt.plot(predictions, color='yellow', label=Filename+' Forecasted price')
    plt.legend()
    plt.show()
