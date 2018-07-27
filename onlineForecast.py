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
parser.add_argument("-pme","--pathModelEntry", type=str, help="Indicate the path to folder containing model. Example : /home/usr/mod/  ", default='./mod')
args = parser.parse_args()
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
        yhat = forecast_lstm(model_b, 1, inte2[-1])

        liste_temps_forec.append(time.time()-t1)

        return yhat



def clear_file(directory_path):

        ''' Delete directory'''

        try:
                shutil.rmtree(directory_path)
        except OSError, e:
                print ("Error: %s - %s." % (e.filename, e.strerror))


def on_press(key):
        try:
                test = 'alphanumeric key {0} pressed'.format(key.char)
        except AttributeError:
                print('special key {0} pressed'.format(key))

def on_release(key):
        #print('{0} released'.format(key))
        if key == keyboard.Key.esc:
                # Stop listener
                stop = True
                return False


def add_image(imag_json, datadict):
        if imag_json.endswith('.json'):

                data = json.load(open("/home/lab3/openpose/output_paulo/json/"+imag_json))
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
	print(test)
	print(liste)
	if test == 0:
		print("you do not move")
	elif test == 1:
		print("you are on your way to the sky")
	elif test == 2:
		print("mayday mayday ! ")



def run(modelPath, erase = True):
                
    ''' modelPath store information about timestep, scaler, interval and
    contains the network model. Standard file :  ~/model.h5 ~/scaler.save ~/tsintr.txt
    erase = True means it erases folder containing json and jpg'''
    
    # erase previous data, openpose output
    if erase:
            clear_file("/home/lab3/openpose/output_paulo/json/")
            clear_file("/home/lab3/openpose/output_paulo/jpg/")
            
            
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


    # run webcam recording+openpose
    os.chdir("/home/lab3/openpose/")
    os.system("./build/examples/openpose/openpose.bin --number_people_max 1 --write_images output_paulo/jpg/ --write_json output_paulo/json/ &") # & to run openpose in background
    nbre_file = 0

    tlj = []
    tla = []
    tld = []
    tlf = []
    openpose = True
    while openpose == True:
	try:
	    t1 = time.time()
            inte = sorted(os.listdir("/home/lab3/openpose/output_paulo/json/"))
	    t12 = time.time()
	    tlj.append(t12-t1)
		
	    
            nbre_file_inte = len(inte)
            if nbre_file_inte > nbre_file:
                    #print(nbre_file, nbre_file_inte, timesteps)
                    nbre_file = nbre_file_inte
		    t1 = time.time()
		    add_image(inte[-1], datadict)
		    t12 = time.time()
		    tla.append(t12-t1)
		    #print(inte[-1])
		    if nbre_file_inte >= timesteps:
			t1 = time.time()
		    	dataf = pd.DataFrame(datadict)
			t12 = time.time()
			tld.append(t12-t1)
			#print(forecast(dataf, timesteps, model, scaler, interval))
			t1 = time.time()
			forec = forecast(dataf, timesteps, model, scaler, interval)
			t12 = time.time()
			tlf.append(t12-t1)
			transfo(forec)
                    #add_image(datadict)
        except:
	    #print("Unexpected error:", sys.exc_info()[0])
	    pass
	
	# break condition of while, stop when openpose is closed	
	try:
	    check_output(["pidof","openpose.bin"])
	except:
	    openpose = False
	    pass

	    


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



    # function forecast is the most time consuming, around 1 sec

    # Display time consumption of function forecast
    

    plt.figure("shape data")
    plt.plot([a for a in range(len(liste_temps_shape_data))], liste_temps_shape_data)
    plt.xlabel("iterations")
    plt.ylabel("time used (s)")

    plt.figure("scale data")
    plt.plot([a for a in range(len(liste_temps_scale))], liste_temps_scale)
    plt.xlabel("iterations")
    plt.ylabel("time used (s)")

    plt.figure("forecast")
    plt.plot([a for a in range(len(liste_temps_forec))], liste_temps_forec)
    plt.xlabel("iterations")
    plt.ylabel("time used (s)")

    plt.show()

                            

            

    


	
	


#res = run("/home/lab3/openpose/output_paulo/json/", "/home/lab3/OP-movReco/mod/")
res = run(PATH_MOD_ENTRY)








