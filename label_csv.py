import pandas as pd



##parser = argparse.ArgumentParser()
##parser.add_argument("pathToCSV", type=str,
##                    help="Indicate the path CSV to label Example : /home/usr/CSV_data/this.csv  ")
##args = parser.parse_args()


##PATH_CSV = args.pathToCSV

PATH_CSV = "/home/rqd/OPlstm/data_video/data_csv/20180725_161246.csv"
PATH_OUTPUT = PATH_CSV[:-4] + "_labeled.csv"


df = pd.read_csv(PATH_CSV, sep=';')

# serie : sit , walk , walk back...    0 up, 1 down

##20180725_160815
##list_stop = [0, 64, 153, 260, 357, 450, 622, 696, 755, 838, 925, 995,
##             1074, 1177, 1271, 1362, 1461, 1556, 1595]

##20180725_161140
##list_stop = [0, 122, 282, 405, 512, 663, 776, 914, 1065, 1154, 1271,
##             1408, 1474]

## 20180725_161246
##list_stop = [0, 129, 246, 387, 636, 809, 935, 1187, 1370,
##             1482, 1597, 1683, 1769]
previous = 0
compteur = 0
compteur2 = 1
for a in range(len(list_stop)-1):
    previous = compteur2 % 3
    for b in range(list_stop[a+1]-list_stop[a]):
        df.at[compteur, 'Unnamed: 0'] = previous
        compteur += 1
        
    
    compteur2 += 1

df.to_csv(PATH_OUTPUT, index=False) # export the labeled data




# serie : arm_up, arm_down, arm up...    0 up,   1 down

# 20180725_155802
##list_stop = [0, 41, 74, 111, 143, 179, 214, 248, 281, 315, 346,
##             372, 394, 416, 462, 509, 526, 542, 559, 578, 595,
##             613, 631, 662, 709, 721] # ne pas oublier de mettre le dernier +1

##20180725_155914
##list_stop = [0, 18, 71, 105, 139, 170, 214, 254, 300, 324, 358, 384,
##             419, 453, 490, 519, 555, 570, 590, 609, 630, 649, 667,
##             682, 697, 711, 729, 745, 765, 790, 811, 834, 860, 883,
##             911, 960, 1018, 1063, 1117, 1163, 1233, 1272, 1328,
##             1369, 1425, 1435]

##20180725_160043
##list_stop = [0, 24, 79, 119, 179, 225, 275, 325, 365, 425, 480,
##             523, 576, 616, 649, 672, 702, 725, 759, 781, 811,
##             834, 862, 885, 910]

##20180725_160137
##list_stop = [0, 17, 77, 130, 183, 222, 249, 271, 296, 320, 351, 409,
##             485, 597, 626, 654, 683, 708, 738, 763, 790, 815, 937,
##             1011, 1051, 1095, 1149, 1198, 1247, 1292, 1335, 1377,
##             1425, 1464, 1517, 1564, 1606, 1653, 1689]
             

##previous = 1
##compteur = 0
##for a in range(len(list_stop)-1):
##    for b in range(list_stop[a+1]-list_stop[a]):
##        df.at[compteur, 'Unnamed: 0'] = previous
##        compteur += 1
##        
##    previous = 1 - previous









