import matplotlib.pyplot as plt
from pandas import read_csv



df = read_csv("/home/rqd/OPlstm/data_csv/d_updown1.csv")

points = df.values


def display_points(listec):
    
    ''' Display points on 2D plan / consider there is none aberrant point '''

    #listec = listec[1:] # label first
    x, y = [], []
    for a in range(len(listec)):
        # confidence, x, y we just want x,y
        if a%3 == 1: # x
            x.append(listec[a])
            
            
        elif a%3 == 2: # y
            y.append(listec[a])
            

    plt.figure('1')
    plt.scatter(x, y, alpha=1)
    plt.show()
    plt.close()

def clean_points(listec):
    
    ''' Delete aberrant point to prevent display flaws '''

    # spot aberrant points
    list_ind_aber = []
    for a in range(len(listec)):
        if a%3 == 1: # coord x
            if listec[a] == 0 and listec[a+1] == 0:
                list_ind_aber.append([a,a+1])

    # "correct" aberrant points, assign it other points mean
    nbre_pts = len(listec)/3
    nbr_aber = len(list_ind_aber)
    mean_x = sum(listc[a] )
    mean_y = sum(listc[a] )

    return(listec)
    
        
