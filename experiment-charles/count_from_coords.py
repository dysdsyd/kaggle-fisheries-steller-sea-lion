# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

coords = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/coords/coords.csv', header=0) 
counts = []

for pictureID in coords.tid.unique():
    #print(pictureID)
    pictureCoords = coords[coords.tid == pictureID]
    countsPicture =  pictureCoords.cls.value_counts()
    for i in range(5):
        try:
            type(countsPicture[i])
        except KeyError:
            countsPicture[i] = 0
    counts.append(countsPicture.sort_index().values.tolist())
#==============================================================================
#     if pictureID ==2:
#         raise Exception("h")
#==============================================================================
df = pd.DataFrame(counts, columns = ["adult_males","subadult_males","adult_females","juveniles","pups"])
df["train_id"] = df.index
df = df[["train_id","adult_males","subadult_males","adult_females","juveniles","pups"]]
df.to_csv("C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/counts/trainCount_v1.csv", index=False)




coords = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/coords/coords_v4.csv', header=0) 
counts = []

for pictureID in coords.tid.unique():
    #print(pictureID)
    pictureCoords = coords[coords.tid == pictureID]
    countsPicture =  pictureCoords.cls.value_counts()
    for i in range(5):
        try:
            type(countsPicture[i])
        except KeyError:
            countsPicture[i] = 0
    counts.append(countsPicture.sort_index().values.tolist())
#==============================================================================
#     if pictureID ==2:
#         raise Exception("h")
#==============================================================================
df = pd.DataFrame(counts, columns = ["adult_males","subadult_males","adult_females","juveniles","pups"])
df["train_id"] = df.index
df = df[["train_id","adult_males","subadult_males","adult_females","juveniles","pups"]]
df.to_csv("C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/counts/trainCount_v4.csv", index=False)
