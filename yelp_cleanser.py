
#function to categorize the review stars
def categorize(x):
    if x>3:return 1
    else: return 0

#function to tag words
count =0
def nounsPOS(POStags,tagger,line):
    global count
    print count
    wordList = []
    line= re.sub('[^a-zA-Z\d]',' ',line)
    line = re.sub(' +',' ',line).strip()
    terms = word_tokenize(line.lower())
    tagged_terms=tagger.tag(terms)
    for pair in tagged_terms:
                for tag in POStags: # for each POS tag
                        if pair[1].startswith(tag): wordList.append(pair[0])
    count +=1
    return wordList
    
import pandas as pd
import numpy as np
import re
from nltk import word_tokenize,load
from geopy.distance import great_circle
from functools import partial

df = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\review.csv")
df1 = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\user.csv")
useful_cols = ["average_stars","user_id","review_count"]
df1 = df1[useful_cols]
df2 = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\business_loc.csv")
df3 = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\user_loc.csv")

df = df.merge(df1,how = 'left',on='user_id')
df = df.merge(df2,how = 'left',on='business_id')
df = df.merge(df3,how = 'left',on='user_id')


df = df[ df['review_count'] > 20 ] #Consider reviews by users with more than 20 reviews
df = df[ df['stars'] != 3 ] #Remove reviews with rating 3
df['stars_category'] = df['stars'].apply(categorize) #categorize the stars

print df.dtypes
print "\n\nBalancing the dataset..."
#Balance the dataset as 50-50
sample_size = len( df[ df['stars_category']==0] )
positive_indices = df[ df['stars_category']==1].index
random_indices = np.random.choice(positive_indices,sample_size,replace=False)
positive_sample = df.loc[random_indices]
negative_sample = df[ df['stars_category']==0 ]
balanced_data = positive_sample.append(negative_sample)

print "\n\nCalculating distance between user and business..."
#Calculate distance between user and business
dist = balanced_data.apply(lambda x: great_circle(x[8], x[9]), axis = 1)
d = []
for i in dist:
    d.append(float((str(i)).split(' ')[0])) #Strip the 'km' string in distance function
balanced_data['dist']=d #Add the dist to the dataframe

print "\n\nPOS tagging..."
#Consider only the nouns in the text
POStags=['NN'] # POS tags of interest
_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle' #make a new tagger
tagger = load(_POS_TAGGER) #Load the tagger

balanced_data['nouns'] = balanced_data['text'].apply( partial(nounsPOS,POStags,tagger) ) #apply the function to all rows of text

"\n\nWriting data to file..."
balanced_data.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\model_data.csv",index=False)


print balanced_data.dtypes
print len(balanced_data[ balanced_data['stars_category']==0 ])
print len(balanced_data[ balanced_data['stars_category']==1 ])
print balanced_data.shape