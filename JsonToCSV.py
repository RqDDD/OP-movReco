import json
from pprint import pprint
import os
import pandas as pd


node=dict()
for i in range(3):
    coord=''
    if i==0:
        coord='_x'
    if i==1:
        coord='_y'
    if i==2:
        coord='_c'
    node["Nose"+coord]=[]
    node["Neck"+coord]=[]
    node["RShoulder"+coord]=[]
    node["RElbow"+coord]=[]
    node["RWrist"+coord]=[]
    node["LShoulder"+coord]=[]
    node["LElbow"+coord]=[]
    node["LWrist"+coord]=[]
    node["RHip"+coord]=[]
    node["RKnee"+coord]=[]
    node["RAnkle"+coord]=[]
    node["LHip"+coord]=[]
    node["LKnee"+coord]=[]
    node["LAnkle"+coord]=[]
    node["REye"+coord]=[]
    node["LEye"+coord]=[]
    node["REar"+coord]=[]
    node["LEar"+coord]=[]

for e in sorted(os.listdir('.')):
    if e.endswith('.json'):
        print(e)
        data = json.load(open(e))
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
            node["Nose"+coord].append(nodes_raw[0+i])
            node["Neck"+coord].append(nodes_raw[3+i])
            node["RShoulder"+coord].append(nodes_raw[6+i])
            node["RElbow"+coord].append(nodes_raw[9+i])
            node["RWrist"+coord].append(nodes_raw[12+i])
            node["LShoulder"+coord].append(nodes_raw[15+i])
            node["LElbow"+coord].append(nodes_raw[18+i])
            node["LWrist"+coord].append(nodes_raw[21+i])
            node["RHip"+coord].append(nodes_raw[24+i])
            node["RKnee"+coord].append(nodes_raw[27+i])
            node["RAnkle"+coord].append(nodes_raw[30+i])
            node["LHip"+coord].append(nodes_raw[33+i])
            node["LKnee"+coord].append(nodes_raw[36+i])
            node["LAnkle"+coord].append(nodes_raw[39+i])
            node["REye"+coord].append(nodes_raw[42+i])
            node["LEye"+coord].append(nodes_raw[45+i])
            node["REar"+coord].append(nodes_raw[48+i])
            node["LEar"+coord].append(nodes_raw[51+i])
            
        
        
        

df=pd.DataFrame(node)
df.to_csv('Data.csv',sep=';')
