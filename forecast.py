from keras.models import load_model
from sklearn.externals import joblib
from testData2 import shape_data
from testData2 import forecast_lstm
from testData2 import indMax
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM


def reconstruct(model, batch_size = 1):

        ''' To overcome the difference between batch_size used for training and for forecasting '''
        
        old_weights = model.get_weights()
        model_b = Sequential()
        inte = model.layers[0].batch_input_shape
        nbr_neurons = model.layers[0].output_shape[2]
        # check the correspondence with "create_model" 
        model_b.add(LSTM(nbr_neurons, batch_input_shape=(batch_size, inte[1], inte[2]), return_sequences=True, stateful=True))
        for a in range(1,len(model.layers)):
                model_b.add(model.layers[a])

        model_b.set_weights(old_weights)
        return(model_b)



def forecast(series, timesteps, model, scaler, interval = 1):

        ''' Forecast some instance with a model pre trained '''
        
        inte1 = shape_data(series, timesteps)
        inte2 = inte1.values[timesteps:-interval,:]

        inte2 = inte2[:, 1:]
        inte2 = scaler.transform(inte2)
        inte2 = inte2.reshape(inte2.shape[0], timesteps+1, int(inte2.shape[1]/(timesteps+1)))

        # batch_size potentially different from learning, here batch_size = 1
        model_b = reconstruct(model, 1)
        
        yhat = []
        for a in range(len(inte2)):
                yhat.append(forecast_lstm(model_b, 1, inte2[a]))
        
        return yhat


def run(dataPath, modelPath):
        
        ''' modelPath store information about timestep, scaler, interval and
        contains the network model. Standard file :  ~/model.h5 ~/scaler.save ~/tsintr.txt'''
        
        # load datasets
        series = read_csv(dataPath)
        
        # loading model, scaler, timestep and interval
        model = load_model(modelPath+'my_model.h5')
        scaler = joblib.load(modelPath + "scaler.save")
        fichier = open(modelPath + "tsintr.txt", "r")
        tsintr = fichier.read().split(";")
        timesteps = int(tsintr[0])
        interval = int(tsintr[1])
        fichier.close()

        predictions = forecast(series, timesteps, model, scaler, interval)
        return(predictions)


res = run("/home/rqd/OPlstm/data_csv/d_updown1.csv", "/home/rqd/OPlstm/mod/")

for a in range(len(res)):
        print(indMax(res[a][0]))
