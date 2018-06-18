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



coord_xy = ['Nose_x', 'Neck_x', 'RShoulder_x', 'RElbow_x', 'RWrist_x', 'LShoulder_x', 'LElbow_x',
 'LWrist_x', 'RHip_x', 'RKnee_x', 'RAnkle_x', 'LHip_x', 'LKnee_x', 'LAnkle_x', 'REye_x',
 'LEye_x', 'REar_x', 'LEar_x', 'Nose_y', 'Neck_y', 'RShoulder_y', 'RElbow_y', 'RWrist_y',
 'LShoulder_y', 'LElbow_y', 'LWrist_y', 'RHip_y', 'RKnee_y', 'RAnkle_y', 'LHip_y', 'LKnee_y',
 'LAnkle_y', 'REye_y', 'LEye_y', 'REar_y', 'LEar_y']

 
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1, interval = 1):
        data = DataFrame(data)
        inte = [[] for a in range(lag)]
        liste_drop = []
        #print(data)
        for a in data.columns[1:]: # first column is label
                if a not in coord_xy:
                        liste_drop.append(a)
                else:
                        data[a] = difference(data[a], interval)
                        for b in range(1, lag+1):
                                inte[b-1].append(data[a].shift(b))
        #print(len(inte[0]), len(inte[1]), len(data.columns[1:]), len(inte), lag)
        #print(data)
        data = data.drop(liste_drop, axis=1)
##        print(liste_drop)
##        print(data.columns[1:])
        # A loop again ; good order ;  not well written
        nbre_car = len(data.columns[1:])
        for a in range(lag): # first column index
                for b in range(nbre_car):
                        #print(a,b)
                        data[str(data.columns[b+1])+'timestep_'+str(a+1)] = inte[a][b]

        #print(data)
        return data
 
# create a differenced series
def difference(dataset, interval=1):
        diff = list()
        #print(dataset)
        for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
        #print(diff)
        return Series(diff)
 
 
# scale train and test data to [-1, 1]
def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        #print(train)

        scaler = scaler.fit(train)

        train = train.reshape(train.shape[0], train.shape[1])

        train_scaled = scaler.transform(train)
        #print(test)
        #print(train_scaled)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

def create_model(structLSTM, structForw, batch_size, shape1, shape2):
        model = Sequential()
        
        model.add(LSTM(structLSTM[0], batch_input_shape=(batch_size, shape1, shape2), return_sequences=True, stateful=True))
        for a in range(len(structLSTM)-1):
                model.add(LSTM(structLSTM[a+1]))
                
        for a in range(len(structForw)-1):
                model.add(Dense(structForw[a]))
        model.add(Dense(structForw[-1]))
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return(model)
 
# fit an LSTM network to training data
def fit_lstm(X_train, y_train, structLSTM, structForw, batch_size, nb_epoch, timesteps):
        
        model = create_model(structLSTM, structForw, batch_size, X_train.shape[1], X_train.shape[2])

        for i in range(nb_epoch):
                model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
                model.reset_states()
        return model
 
# make a one-step forecast
def forecast_lstm(old_weights, structLSTM, structForw, batch_size, X):
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

        inte1 = timeseries_to_supervised(series[0], timesteps)
        inte2 = inte1.values[timesteps:-interval,:]
        supervised_values = inte2
        for a in range(1,len(series)):
                inte1 = timeseries_to_supervised(series[a], timesteps)
                inte2 = inte1.values[timesteps:-interval,:]
                supervised_values = np.concatenate((supervised_values, inte2))



        # Put shuffle here

        liste_alea = np.arange(len(supervised_values))

        np.random.shuffle(liste_alea)



        # split data into train and test-sets
        nbre_aprent = int(len(supervised_values)/100*80)
        #train, test = supervised_values[0:nbre_aprent,:], supervised_values[nbre_aprent:-interval,:]
        train = np.array([supervised_values[a] for a in liste_alea[0:nbre_aprent]])
        test = np.array([supervised_values[a] for a in liste_alea[nbre_aprent:]])
        print(train)

        #Add data taking account of depth


        #Add data with noise
        train = transformation.addNoise(train, 300, 5)
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

        X_scaled = X_scaled.reshape(X.shape[0], timesteps+1, int(X.shape[1]/(timesteps+1))) # attention au timestep ici
        Xt_scaled = Xt_scaled.reshape(Xt.shape[0], timesteps+1, int(Xt.shape[1]/(timesteps+1))) # attention au timestep ici

        error_scores = list()
        for r in range(repeats):
                # fit the base model
                global lstm_model
                structLSMT = [100,50]
                structForw = [30,10,3]
                lstm_model = fit_lstm(X_scaled, y, structLSMT, structForw,  20, 40, timesteps)
                old_weights = lstm_model.get_weights()
                
                # forecast test dataset
                global predictions
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
 
# execute the experiment
def run(listeChemins, pathwd): #liste chemins d'apprentissage
        # load datasets
        series = []
        os.chdir(pathwd)
        
        for a in listeChemins:
                series.append(read_csv(a))
        #series = read_csv("/home/rqd/Documents/NORMbittrex-BTCUSDT-1d.csv",index_col=0)
        # experiment
        repeats = 1
        results = DataFrame()
        # run experiment
        timesteps = 10
        results['results'] = experiment(repeats, series, timesteps)
        # summarize results
        #print(results.describe())
        # save results
        #results.to_csv('experiment_timesteps_1.csv', index=False)



 
 # entry point
print([e for e in os.listdir('/home/rqd/OPlstm/data_csv')])
run([e for e in os.listdir('/home/rqd/OPlstm/data_csv')], '/home/rqd/OPlstm/data_csv')



# different batch size training/test
# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/




# Bien saisir nuance entre interval et timestep
