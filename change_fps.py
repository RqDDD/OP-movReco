import pandas as pd
import numpy as np
import os




## Delete some line to simulate lowering fps, supposed to be already labeled data
## We could also consider adding some interpolated line to simulate increasing fps 

PATH_ENTRY = "/home/rqd/OPlstm/data_csv/"
PATH_OUTPUT = PATH_ENTRY



def decrease_fps(path, percentDecrease = 20):
    
    ''' Delete a given percent of data lines from the original and save it in a new csv '''

    liste_files = os.listdir(path)
    for e in liste_files:
        df = pd.read_csv(path + e)
        n= len(df)
        index_deleted = np.int0(np.linspace(0, n , int(percentDecrease/100*n), endpoint=False)) # not beautiful at all
        print(e, index_deleted)
        df.drop(df.index[index_deleted])
        df.to_csv(PATH_OUTPUT + e[:-4] + "_" + "decFPS" + str(percentDecrease) + ".csv", index=False) # export the new data 


decrease_fps(PATH_ENTRY, percentDecrease = 20)


def test_decrease_fps(path_to_test):
    PATH_CSV = "/home/rqd/OPlstm/data_csv/d_updown1.csv"
    df = read_csv(path_to_test)
