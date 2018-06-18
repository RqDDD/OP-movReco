from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy as np
import os

import transformation


# Liste of useful column
coord_xy = ['Nose_x', 'Neck_x', 'RShoulder_x', 'RElbow_x', 'RWrist_x', 'LShoulder_x', 'LElbow_x',
 'LWrist_x', 'RHip_x', 'RKnee_x', 'RAnkle_x', 'LHip_x', 'LKnee_x', 'LAnkle_x', 'REye_x',
 'LEye_x', 'REar_x', 'LEar_x', 'Nose_y', 'Neck_y', 'RShoulder_y', 'RElbow_y', 'RWrist_y',
 'LShoulder_y', 'LElbow_y', 'LWrist_y', 'RHip_y', 'RKnee_y', 'RAnkle_y', 'LHip_y', 'LKnee_y',
 'LAnkle_y', 'REye_y', 'LEye_y', 'REar_y', 'LEar_y']

 
 
# frame a sequence as a supervised learning problem
def shape_data(data, lag=1, interval = 1):
        
        '''First, compute the difference between line : interval between which line difference is computed
        and then concatenate different line to "create" the timestep : lag = number of timesteps'''
        
        data = DataFrame(data)
        inte = [[] for a in range(lag)]

        # useless column
        liste_drop = []
        
        for a in data.columns[1:]: # first column is label
                if a not in coord_xy:
                        liste_drop.append(a)
                else:
                        data[a] = difference(data[a], interval)
                        for b in range(1, lag+1):
                                inte[b-1].append(data[a].shift(b))

        # remove useless columns from the dataframe
        data = data.drop(liste_drop, axis=1)
        

        # A loop again ; to garantee good order (time ordrered) ;  not well written
        nbre_car = len(data.columns[1:])
        for a in range(lag): # first column index
                for b in range(nbre_car):                 
                        data[str(data.columns[b+1])+'timestep_'+str(a+1)] = inte[a][b]
    
        return data
 

def difference(dataset, interval=1):
        
        ''' create a differenced series '''
        
        diff = list()
        for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
        
        return Series(diff)
 
 

def scale(train, test):
        
        ''' scale train and test data to [-1, 1] '''
        
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)

        train = train.reshape(train.shape[0], train.shape[1])

        train_scaled = scaler.transform(train)

        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

def create_model(structLSTM, structForw, batch_size, shape1, shape2):
        
        ''' Create partly LSTM model :  structLSTM = [number of neurons in the first layer, in the second,...]
        structForw = [..., ... , number of neurons in the last layer : correspond to the number of class]'''

        
        model = Sequential()
        # first layer
        model.add(LSTM(structLSTM[0], batch_input_shape=(batch_size, shape1, shape2), return_sequences=True, stateful=True))
        
        # add the hidden layer of lstm part
        for a in range(len(structLSTM)-1):
                model.add(LSTM(structLSTM[a+1]))
                
        # add the hidden layer of forward part        
        for a in range(len(structForw)-1):
                model.add(Dense(structForw[a]))

        # output layer
        model.add(Dense(structForw[-1]))
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return(model)
 

def fit_lstm(X_train, y_train, structLSTM, structForw, batch_size, nb_epoch, timesteps):

        ''' fit the network to the training data '''
        
        model = create_model(structLSTM, structForw, batch_size, X_train.shape[1], X_train.shape[2])

        for i in range(nb_epoch):
                model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
                model.reset_states()
        return model
 
# make a one-step forecast
def forecast_lstm(old_weights, structLSTM, structForw, batch_size, X):

        ''' forecast output with X as entry. The batch size is potentially different from the one
        used for training'''
        
        X = X.reshape(1, X.shape[0], X.shape[1])
        model_b = create_model(structLSTM, structForw, batch_size, X.shape[1], X.shape[2])
        model_b.set_weights(old_weights)
        yhat = model_b.predict(X, batch_size=batch_size)
        return yhat

def score(yt, prediction):
        accuracy = 0
        n = len(yt)
        for a in range(n):
                if indMax(yt[a]) == indMax(prediction[a][0]):
                        accuracy += 1
        accuracy = accuracy/n*100
        return(accuracy)

def indMax(liste):
        maxi = liste[0]
        ind = 0
        for a in range(1, len(liste)):
                if liste[a] > maxi:
                        maxi = liste[a]
                        ind = a
        return(ind)
        
# run a repeated experiment
def experiment(repeats, series, timesteps, interval = 1):

        inte1 = shape_data(series[0], timesteps)
        inte2 = inte1.values[timesteps:-interval,:]
        supervised_values = inte2
        for a in range(1,len(series)):
                inte1 = shape_data(series[a], timesteps)
                inte2 = inte1.values[timesteps:-interval,:]
                supervised_values = np.concatenate((supervised_values, inte2))



        # split data into train and test-sets
        
        # shuffle here
        liste_alea = np.arange(len(supervised_values))
        np.random.shuffle(liste_alea)
        
        nbre_aprent = int(len(supervised_values)/100*80)
        train = np.array([supervised_values[a] for a in liste_alea[0:nbre_aprent]])
        test = np.array([supervised_values[a] for a in liste_alea[nbre_aprent:]])
        print(train)
        

        #Add data taking account of depth


        #Add data with noise
        train = transformation.addNoise(train, 300, 5)

        # shuffle / to avoid finish training with spoiled data
        np.random.shuffle(train)        
        
        X, label = train[:, 1:], train[:, 0]
        Xt, labelt = test[:, 1:], test[:, 0]
        
        # transform the scale of the data
        scaler, X_scaled, Xt_scaled = scale(X, Xt)
        
        # One hot encoding label
        onehot_encoder = OneHotEncoder(sparse=False)
        y = label.reshape(len(label), 1)
        y = onehot_encoder.fit_transform(y)
        yt = labelt.reshape(len(labelt), 1)
        yt = onehot_encoder.transform(yt)

        X_scaled = X_scaled.reshape(X.shape[0], timesteps+1, int(X.shape[1]/(timesteps+1))) # mind the timestep here
        Xt_scaled = Xt_scaled.reshape(Xt.shape[0], timesteps+1, int(Xt.shape[1]/(timesteps+1))) # mind the timestep here

        error_scores = list()
        for r in range(repeats):
                # fit the base model / test different models on the cloud
                structLSMT = [100,50]
                structForw = [30,10,3]
                lstm_model = fit_lstm(X_scaled, y, structLSMT, structForw,  20, 40, timesteps)
                old_weights = lstm_model.get_weights()
                
                # forecast test dataset    
                predictions = list()
                for i in range(len(Xt)):
                        # predict
                        yhat = forecast_lstm(old_weights, structLSMT, structForw, 1, Xt_scaled[i])
                        # store forecast
                        predictions.append(yhat)
                        
                # report performance
                for a in range(len(yt)):
                        print(yt[a], predictions[a][0])
                        print(a)
                print(" Accuracy : ", score(yt,predictions) )
                

        return error_scores
 

def run(listeChemins, pathwd): #liste chemins d'apprentissage
        
        ''' Execute the experiment'''
        
        # load datasets
        series = []
        # Current directory
        os.chdir(pathwd)
        
        for a in listeChemins:
                series.append(read_csv(a))
        
        # number of repeats / a bit redundants with nb_epoch
        repeats = 1   

        timesteps = 10       
        
        experiment(repeats, series, timesteps)

        results = DataFrame()
        #results['results'] = experiment(repeats, series, timesteps)




 
 # entry point
print([e for e in os.listdir('/home/rqd/OPlstm/data_csv')])
run([e for e in os.listdir('/home/rqd/OPlstm/data_csv')], '/home/rqd/OPlstm/data_csv')



# different batch size training/test
# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/




# Bien saisir nuance entre interval et timestep
