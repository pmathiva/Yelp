# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 07:45:36 2016

@author: priya.cse2009
"""
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import numpy as np
from collections import Counter

df1 = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\review.csv")
df2 = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\business.csv")
df = df1.merge(df2, on='business_id')
#tmp = df[0:5]
#print len(tmp['user_id'].unique())


def getUserLoc(x):
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    cols = ['latitude','longitude']
    coords = x[cols]
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db = db.fit(np.radians(coords))
    most_dense_cluster = Counter(db.labels_).most_common(1)[0][0]
    cluster =   coords [db.labels_ == most_dense_cluster] 
    cluster = zip(cluster['latitude'],cluster['longitude'])
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    return centroid

groups = df.groupby('user_id')

userLocList = []
for name,grp in groups:
    loc = getUserLoc(grp)
    val = (name,loc)
    userLocList.append(val)
    

user_loc_df = pd.DataFrame.from_records(userLocList)
user_loc_df.columns = ['user_id', 'loc']
print user_loc_df.shape
print user_loc_df.dtypes
print len(df  ['user_id'].unique())
user_loc_df.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\user_loc.csv", index = False)