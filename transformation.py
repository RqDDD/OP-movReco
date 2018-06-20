import numpy as np

def simDepth(addPerCent, listeCoef = np.arange(0,3,0.5)):
    databis = data.copy()
    n = databis.shape[1] # 2 dimensionnal array
    nl = databis.shape[0]
    # add artificial sample len(sampleAdded) = len(InitialSample) * addPerCent//100
    for a in range(addPerCent//100):
        for b in range(nl):
            inte = listeCoef[b%b]
            inte2 = databis[b]/inte
        databis = np.concatenate((databis, inte2))
        

    rest = int(addPerCent - 100*(addPerCent//100))
    if rest !=0:
        liste_alea = np.arange(len(data))
        np.random.shuffle(liste_alea)
        nbre_ref = int(len(data)/100*rest)+1       
        inte = data.copy()
        val_ref = np.array([inte[a] for a in liste_alea[0:int(nbre_ref/len(listeCoef))]])
        for a in listeCoef:
            for b in range(1,n): # first column = label
                val_ref[:,b] = val_ref[:,b]/a
            databis = np.concatenate((databis, val_ref))

    return(databis)



def addNoise(data, addPerCent, stdPerCent): # ajout de ligne en pourcent peut être >100 / add random gaussian noise
    '''Add a gaussian noise to each column to artificially inflate
    the learning sample and eventually improve a learning process
    mean=std(column) sigma=1'''
    sigma = 1
    databis = data.copy()
    n = databis.shape[1] # 2 dimensionnal array
    nl = databis.shape[0]
    # add artificial sample len(sampleAdded) = len(InitialSample) * addPerCent//100
    for a in range(addPerCent//100):
        inte = data.copy()
        for b in range(1,n): # first column = label
            mu = np.std(inte[:,b]) * stdPerCent/100
            inte[:,b] = inte[:,b] + (sigma*np.random.randn(nl) + mu)
        databis = np.concatenate((databis, inte))

        

    rest = int(addPerCent - 100*(addPerCent//100))
    print(addPerCent)
    print(rest)
    if rest !=0:
        liste_alea = np.arange(len(data))
        np.random.shuffle(liste_alea)
        nbre_ref = int(len(data)/100*rest)+1
        
        inte = data.copy()
        val_ref = np.array([inte[a] for a in liste_alea[0:nbre_ref]])
        nla = len(val_ref)
        for b in range(1,n): # first column = label
            mu = np.std(val_ref[:,b]) * stdPerCent/100 
            val_ref[:,b] = val_ref[:,b] + (sigma*np.random.randn(nla) + mu)
        databis = np.concatenate((databis, val_ref))

    return(databis)
            

#test = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
#gg = addNoise(test, 50, 10)
    
