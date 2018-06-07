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




 
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1, interval = 1):
        data = DataFrame(data)
        inte = [[] for a in range(lag)]
        #print(data)
        for a in data.columns[1:]: # first column is label
                data[a] = difference(data[a], interval)
                for b in range(1, lag+1):
                        inte[b-1].append(data[a].shift(b))
        #print(len(inte[0]), len(inte[1]), len(data.columns[1:]), len(inte), lag)
                        
        # A loop again ; good order ;  not well written
        nbre_car = len(data.columns[1:])
        for a in range(lag): # first column index
                for b in range(nbre_car):
                        #print(a,b)
                        data[str(data.columns[b+1])+'timestep_'+str(a+1)] = inte[a][b]

        print(data)
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
 
 
# fit an LSTM network to training data
def fit_lstm(X_train, y_train, batch_size, nb_epoch, neurons, timesteps):

        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]), return_sequences=True, stateful=True))
        model.add(LSTM(50))
        model.add(Dense(10))
        model.add(Dense(10))
        model.add(Dense(3))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
                model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
                model.reset_states()
        return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
        X = X.reshape(1, X.shape[0], X.shape[1])
        yhat = model.predict(X, batch_size=batch_size)
        return yhat

def score(yt, prediction):
        accuracy = 0
        n = len(yt)
        for a in range(n):
##                print("EXACT")
##                print(indMax(yt[a]))
##                print(yt[a], "\n")
##                print("PRED")
##                print(indMax(prediction[a][0]))
##                print(prediction[a][0], "\n", "\n")
                if indMax(yt[a]) == indMax(prediction[a][0]):
                        accuracy += 1
        accuracy = accuracy/n*100
        return(accuracy)

def indMax(liste):
        maxi = liste[0]
        ind = 0
        for a in range(1, len(liste)):
                if liste[a] > maxi:
                        mini = liste[a]
                        ind = a
        return(ind)
        
# run a repeated experiment
def experiment(repeats, series, timesteps, interval = 1):
        #print(raw_values)
        #diff_values = difference(raw_values, 1)
        #print(diff_values)
        # transform data to be supervised learning
        inte1 = timeseries_to_supervised(series[0], timesteps)
        inte2 = inte1.values[timesteps:-interval,:]
        supervised_values = inte2
        for a in range(1,len(series)):
                inte1 = timeseries_to_supervised(series[a], timesteps)
                inte2 = inte1.values[timesteps:-interval,:]
                supervised_values = np.concatenate((supervised_values, inte2))


        print(supervised_values)



        # Put shuffle here

        liste_alea = np.arange(len(supervised_values))

        np.random.shuffle(liste_alea)



        # split data into train and test-sets
        nbre_aprent = int(len(supervised_values)/100*80)
        #train, test = supervised_values[0:nbre_aprent,:], supervised_values[nbre_aprent:-interval,:]
        train = np.array([supervised_values[a] for a in liste_alea[0:nbre_aprent]])
        test = np.array([supervised_values[a] for a in liste_alea[nbre_aprent:]])
        #print(train)


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

        X = X.reshape(X.shape[0], timesteps+1, int(X.shape[1]/(timesteps+1))) # attention au timestep ici
        Xt = Xt.reshape(Xt.shape[0], timesteps+1, int(Xt.shape[1]/(timesteps+1))) # attention au timestep ici
        #print(train_scaled)
        #print(scaler)
        # run experiment
        #print(X[15])
        error_scores = list()
        for r in range(repeats):
                # fit the base model
                global lstm_model
                lstm_model = fit_lstm(X, y, 1, 200, 100, timesteps)
                # forecast test dataset
                global predictions
                predictions = list()
                for i in range(len(Xt)):
                        # predict
                        yhat = forecast_lstm(lstm_model, 1, Xt[i])
                        # store forecast
                        predictions.append(yhat)
                # report performance
                print(yt)
                print(predictions)
                print(" Accuracy : ", score(yt,predictions) )
                #rmse = sqrt(mean_squared_error(yt, predictions))
                #print('%d) Test RMSE: %.3f' % (r+1, rmse))
                #error_scores.append(rmse)
        return error_scores
 
# execute the experiment
def run(listeChemins): #liste chemins d'apprentissage
        # load datasets
        series = []
        for a in listeChemins:
                series.append(read_csv(a))
        #series = read_csv("/home/rqd/Documents/NORMbittrex-BTCUSDT-1d.csv",index_col=0)
        # experiment
        repeats = 1
        results = DataFrame()
        # run experiment
        timesteps = 2
        results['results'] = experiment(repeats, series, timesteps)
        # summarize results
        #print(results.describe())
        # save results
        #results.to_csv('experiment_timesteps_1.csv', index=False)
 
 # entry point
run(['/home/rqd/Documents/StagePortu/DataOpenPosetronq.csv','/home/rqd/Documents/StagePortu/DataOpenPosetronq2.csv'])








# Bien saisir nuance entre interval et timestep
