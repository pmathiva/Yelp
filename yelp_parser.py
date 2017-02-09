# -*- coding: utf-8 -*-
import re
import pandas as pd
import io



def makeBusinessTable(line,businessList):        
        myDict = {}
        bid = re.findall( '\"(business_id)\": \"(.*?)\"' , line)
        name = re.findall( '\"(name)\": \"(.*?)\"' , line)
        addr = re.findall('\"(full_address)\": \"(.*?)\"' , line)
        hrs = re.findall('\"(hours)\": {(.*?)}}',line)
        neigh = re.findall('\"(neighborhoods)\": (\[.*?\])',line)
        city = re.findall('\"(city)\": \"(.*?)\"',line)
        state = re.findall('\"(state)\": \"(.*?)\"',line)
        latitude = re.findall('\"(latitude)\": (.*?),',line)
        longitude = re.findall('\"(longitude)\": (.*?),',line)
        stars = re.findall('\"(stars)\": (.*?),',line)
        review_count = re.findall('\"(review_count)\": (.*?),',line)
        categories = re.findall('\"(categories)\": (\[.*?\])',line)
        attr = re.findall('\"(attributes)\": (.*), \"type\"',line)
        opn =  re.findall('\"(open)\": ([a-z]+)',line)
        tokens = [bid,name,addr,hrs,neigh,city,state,latitude,longitude,stars,review_count,categories,opn,attr]
        for t in tokens:
                if t:
                    k , v = t[0][0] , t[0][1]
                    myDict[k] = v 
        businessList.append(myDict)

        
def makeReviewsTable(line,reviewList):          
            myDict = {}
            bid = re.findall( '\"(business_id)\": \"(.*?)\"' ,line)
            uid = re.findall( '\"(user_id)\": \"(.*?)\"' , line)
            stars = re.findall( '\"(stars)\": (.*?),' , line)
            text = re.findall( '\"(text)\": \"(.*?)\"' , line)
            date = re.findall( '\"(date)\": \"(.*?)\"' , line)
            votes = re.findall( '\"(votes)\": {(.*?)}' , line)
            tokens = [bid,uid,stars,text,date,votes]
            for t in tokens:
                if t:
                    k , v = t[0][0] , t[0][1]
                    myDict[k] = v
            reviewList.append(myDict)  
    

def makeUserTable(line,userList):
        myDict = {}
        uid = re.findall( '\"(user_id)\": \"(.*?)\"' , line)
        name = re.findall( '\"(name)\": \"(.*?)\"' , line)
        review_count = re.findall('\"(review_count)\": (.*?),',line)
        averagestars = re.findall('\"(average_stars)\": (.*?),',line)
        votes = re.findall('\"(votes)\": {(.*?)}',line)
        friends = re.findall('\"(friends)\": \[(.*?)\]',line)
        elite = re.findall('\"(elite)\": \[(.*?)\]',line)
        yelpingsince = re.findall('\"(yelping_since)\": \"(.*?)\"',line)
        compliments = re.findall('\"(compliments)\": {(.*?)}',line)
        fans = re.findall('\"(fans)\": (.*?),',line)
        tokens = [uid,name,review_count,averagestars,votes,friends,elite,yelpingsince,compliments,fans]
        for t in tokens:
                if t:
                    k , v = t[0][0] , t[0][1]
                    myDict[k] = v 
        userList.append(myDict)

    
def makeTable(fname):
    businessList = []
    reviewList = []
    userList = []
   
    myFile = io.open(fname,'r',encoding="utf-8",errors='ignore')
    for line in myFile:
        line = line.encode('utf-8')
        ytype = re.findall( '"type": "(.*?)"' , line)
        if ytype:
            if ytype[0] =="business":makeBusinessTable(line,businessList)
            elif ytype[0] == "review":makeReviewsTable(line,reviewList)
            elif ytype[0] == "user":makeUserTable(line,userList)
         
        
    df_business = pd.DataFrame(businessList)
    df_business.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\business.csv",index = False)
    print df_business.shape
    
    df_rev = pd.DataFrame(reviewList)
    df_rev.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\review.csv",index = False)
    print df_rev.shape
    
    df_user = pd.DataFrame(userList)
    df_user.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\user.csv",index = False)
    print df_user.shape
    
    
makeTable("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\yelp_dataset_challenge_academic_dataset")





