from keras.models import load_model
from sklearn.externals import joblib
from testData2 import shape_data
from testData2 import forecast_lstm
from testData2 import indMax
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM
import os
import shutil
from pynput import keyboard
import time
import pandas as pd
import json
import sys
import matplotlib.pyplot as plt
from subprocess import check_output


parser = argparse.ArgumentParser()
parser.add_argument("pathToJSON", type=str,
                    help="Indicate the path to folder containing JSON : obtained running openpose on a video. Example : /home/usr/resultJSON/  ")
parser.add_argument("-pme","--pathModelEntry", type=str,
                    help="Indicate the path to folder containing model. Example : /home/usr/mod/  ", default='./mod')
args = parser.parse_args()
PATH_JSON = args.pathToJSON
PATH_MOD_ENTRY = args.pathEntry

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

global liste_temps_shape_data
liste_temps_shape_data = []

global liste_temps_scale
liste_temps_scale = []

global liste_temps_forec
liste_temps_forec = []

def forecast(series, timesteps, model, scaler, interval = 1):

        ''' Forecast some instance with a model pre trained '''


        t1 = time.time()	
        inte1 = shape_data(series, timesteps)
        liste_temps_shape_data.append(time.time()-t1)

        inte2 = inte1.values[timesteps:-interval,:]
        inte2 = inte2[:, 1:]
        t1 = time.time()
        inte2 = scaler.transform(inte2)
        liste_temps_scale.append(time.time()-t1)

        inte2 = inte2.reshape(inte2.shape[0], timesteps+1, int(inte2.shape[1]/(timesteps+1)))

        # batch_size potentially different from learning, here batch_size = 1
        model_b = reconstruct(model, 1)

        # forecast the last 
        t1 = time.time()
        yhat = []
        for vec in inte2:
                yhat.append(forecast_lstm(model_b, 1, vec))

        liste_temps_forec.append(time.time()-t1)

        return yhat



def clear_file(directory_path):

        ''' Delete directory'''

        try:
                shutil.rmtree(directory_path)
        except (OSError, e):
                print ("Error: %s - %s." % (e.filename, e.strerror))




def add_image(imag_json, datadict):
        if imag_json.endswith('.json'):

                data = json.load(open(PATH_JSON + imag_json))
                #pprint(data["maps"][0]["id"])
                nodes_raw=data["people"][0]["pose_keypoints_2d"]

                for i in range(3): 
                        coord=''
                        if i==0:
                                coord='_x'
                        if i==1:
                                coord='_y'
                        if i==2:
                                coord='_c'
                        # a loop could be done, clearer this way
                        datadict["Nose"+coord].append(nodes_raw[0+i])
                        datadict["Neck"+coord].append(nodes_raw[3+i])
                        datadict["RShoulder"+coord].append(nodes_raw[6+i])
                        datadict["RElbow"+coord].append(nodes_raw[9+i])
                        datadict["RWrist"+coord].append(nodes_raw[12+i])
                        datadict["LShoulder"+coord].append(nodes_raw[15+i])
                        datadict["LElbow"+coord].append(nodes_raw[18+i])
                        datadict["LWrist"+coord].append(nodes_raw[21+i])
                        datadict["RHip"+coord].append(nodes_raw[24+i])
                        datadict["RKnee"+coord].append(nodes_raw[27+i])
                        datadict["RAnkle"+coord].append(nodes_raw[30+i])
                        datadict["LHip"+coord].append(nodes_raw[33+i])
                        datadict["LKnee"+coord].append(nodes_raw[36+i])
                        datadict["LAnkle"+coord].append(nodes_raw[39+i])
                        datadict["REye"+coord].append(nodes_raw[42+i])
                        datadict["LEye"+coord].append(nodes_raw[45+i])
                        datadict["REar"+coord].append(nodes_raw[48+i])
                        datadict["LEar"+coord].append(nodes_raw[51+i])
                        
        return(datadict)

def transfo(liste):
	test = indMax(liste[0])
	if test == 0:
		print("STOP")
	elif test == 1:
		print("UP")
	elif test == 2:
		print("DOWN ! ")

def count_move(liste):
	previous_move = "nonem"
	previous_frame = "nonef"
	compteur_updown = 0
	sustent = 0
	for a in range(len(liste)):
		inte = indMax(liste[a][0]) 
		transfo(liste[a])
		if inte == 0 and inte != previous_frame:
			sustent = 1
			previous_move = previous_frame

		if sustent == 1 and inte != previous_move and inte != 0:
			compteur_updown += 1
			sustent = 0			

		previous_frame = inte

	print(compteur_updown)


def count_move2(liste):
	previous_frame = "nonef"
	compteur_updown = 0
	sustent = 0
	for a in range(len(liste)-1):
		if previous_frame != 0 and indMax(liste[a][0]) == 0 and indMax(liste[a+1][0]) == 0 :
			compteur_updown +=1

		previous_frame = indMax(liste[a][0])

	compteur_updown = (compteur_updown-1)/2
			

	print(compteur_updown)
			
			



def run(modelPath, erase = True):
                
    ''' modelPath store information about timestep, scaler, interval and
    contains the network model. Standard file :  ~/model.h5 ~/scaler.save ~/tsintr.txt
    erase = True means it erases folder containing json and jpg '''
          
    # loading model, scaler, timestep and interval
    model = load_model(modelPath+'my_model.h5')
    scaler = joblib.load(modelPath + "scaler.save")
    fichier = open(modelPath + "tsintr.txt", "r")
    tsintr = fichier.read().split(";")
    timesteps = int(tsintr[0])
    interval = int(tsintr[1])
    fichier.close()
    


    #Initialize the data
    datadict=dict()
    for i in range(3):
            coord=''
            if i==0:
                    coord='_x'
            if i==1:
                    coord='_y'
            if i==2:
                    coord='_c'
            datadict["Nose"+coord]=[]
            datadict["Neck"+coord]=[]
            datadict["RShoulder"+coord]=[]
            datadict["RElbow"+coord]=[]
            datadict["RWrist"+coord]=[]
            datadict["LShoulder"+coord]=[]
            datadict["LElbow"+coord]=[]
            datadict["LWrist"+coord]=[]
            datadict["RHip"+coord]=[]
            datadict["RKnee"+coord]=[]
            datadict["RAnkle"+coord]=[]
            datadict["LHip"+coord]=[]
            datadict["LKnee"+coord]=[]
            datadict["LAnkle"+coord]=[]
            datadict["REye"+coord]=[]
            datadict["LEye"+coord]=[]
            datadict["REar"+coord]=[]
            datadict["LEar"+coord]=[]



    nbre_file = 0

    tlj = []
    tla = []
    tld = []
    tlf = []
    

	
    t1 = time.time()
    inte = sorted(os.listdir(PATH_JSON))
    t12 = time.time()
    tlj.append(t12-t1)
	
    
    nbre_file_inte = len(inte)
    
    #print(inte[-1])
    if nbre_file_inte >= timesteps:
        for img in inte:
                add_image(img, datadict)


        t1 = time.time()
        dataf = pd.DataFrame(datadict)
        t12 = time.time()
        tld.append(t12-t1)
        #print(forecast(dataf, timesteps, model, scaler, interval))
        t1 = time.time()
        forec = forecast(dataf, timesteps, model, scaler, interval)
        t12 = time.time()
        tlf.append(t12-t1)
        count_move2(forec)
    #add_image(datadict)


	    


    # Display time consumption


    #plt.figure("manipulate json")
    #plt.plot([a for a in range(len(tlj))], tlj)
    #plt.xlabel("iterations")
    #plt.ylabel("time used (s)")

    #plt.figure("add image points")
    #plt.plot([a for a in range(len(tla))], tla)
    #plt.xlabel("iterations")
    #plt.ylabel("time used (s)")

    #plt.figure("transform to dataframe")
    #plt.plot([a for a in range(len(tld))], tld)
    #plt.xlabel("iterations")
    #plt.ylabel("time used (s)")

    #plt.figure("function forecast ; difference / timestep / forecasting")
   # plt.plot([a for a in range(len(tlf))], tlf)
  #  plt.xlabel("iterations")
 #   plt.ylabel("time used (s)")

	
   # plt.show()



                            

            

    


	
	


#res = run("/home/lab3/openpose/output_paulo/json/", "/home/lab3/OP-movReco/mod/")
res = run("/home/lab3/OP-movReco/mod/")








