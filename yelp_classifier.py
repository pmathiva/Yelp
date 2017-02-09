


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


balanced_data = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\model_data.csv")


"""
--------------------------------Classifier-------------------------------------------

"""
print "\n\nRunning classifiers..."
#split dataset into training and testing 70-30
train, test = train_test_split(balanced_data, test_size = 0.3)

rev_train  = train['nouns']  
rev_test = test['nouns']

labels_train = train['stars_category']
labels_test = test['stars_category']

#Build a counter based on the training dataset
counter = CountVectorizer(min_df=1000)
counter.fit(rev_train) #Learn a vocabulary dictionary of all tokens from the nouns of training set.
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)##transform the testing data

#build the parameter grid
param_grid1 = [
  {'alpha': [0.1,0.5,1.0]}
]

param_grid2 = [
  {'C': [0.1,0.5,1.0]}
]

param_grid3 = [
  {'n_estimators': [5,10,15],'criterion':['gini','entropy']}
]

#build a grid search to find the best parameters
clf1 = GridSearchCV(MultinomialNB(), param_grid1, cv=5)
clf2 = GridSearchCV(LogisticRegression(), param_grid2, cv=5)
clf3 = GridSearchCV(RandomForestClassifier(), param_grid3, cv=5)
eclf = VotingClassifier(estimators=[('mnb', clf1), ('lr', clf2), ('rf', clf3)], voting='hard')


#---------Run classifier based only on nouns in the review text
clf1.fit(counts_train,labels_train)     #train on nouns in train set
clf2.fit(counts_train,labels_train)     #train on nouns in train set
clf3.fit(counts_train,labels_train)     #train on nouns in train set
eclf.fit(counts_train,labels_train)     #train on nouns in train set

pred1=clf1.predict(counts_test)			#predict on nouns in test set
pred2=clf2.predict(counts_test)			#predict on nouns in test set
pred3=clf3.predict(counts_test)			#predict on nouns in test set
pred=eclf.predict(counts_test)			#predict on nouns in test set

print "\nAccuracy of classifiers using only nouns\n"
print "Multinomial_NB : ",accuracy_score(pred1,labels_test) #print accuracy
print "Logistic  : ",accuracy_score(pred2,labels_test) #print accuracy
print "Randomforest : ",accuracy_score(pred3,labels_test) #print accuracy
print "Voting : ",accuracy_score(pred,labels_test) #print accuracy

print "\nBest parameters",clf1.best_params_
print "Best parameters",clf2.best_params_
print "Best parameters",clf3.best_params_

#-------Run classifier based only on average stars of user
cols = ['average_stars']
train_data_avgstar = train[cols]
test_data_avgstar = test[cols]

clf1.fit(train_data_avgstar,labels_train)     #train on nouns in train set
clf2.fit(train_data_avgstar,labels_train)     #train on nouns in train set
clf3.fit(train_data_avgstar,labels_train)     #train on nouns in train set
eclf.fit(train_data_avgstar,labels_train)     #train on nouns in train set

pred1=clf1.predict(test_data_avgstar)			#predict on nouns in test set
pred2=clf2.predict(test_data_avgstar)			#predict on nouns in test set
pred3=clf3.predict(test_data_avgstar)			#predict on nouns in test set
pred=eclf.predict(test_data_avgstar)			#predict on nouns in test set

print "\nAccuracy of classifiers using only average stars\n"
print "Multinomial_NB : ",accuracy_score(pred1,labels_test) #print accuracy
print "Logistic  : ",accuracy_score(pred2,labels_test) #print accuracy
print "Randomforest : ",accuracy_score(pred3,labels_test) #print accuracy
print "Voting : ",accuracy_score(pred,labels_test) #print accuracy

print "\nBest parameters",clf1.best_params_
print "Best parameters",clf2.best_params_
print "Best parameters",clf3.best_params_

#---------Run classifier based on nouns and average_stars of user
cols = ['average_stars']
train_data = np.concatenate((counts_train.toarray(), train[cols].as_matrix()), axis=1) 
test_data = np.concatenate((counts_test.toarray(), test[cols].as_matrix()), axis=1)

clf1.fit(train_data,labels_train)     #train on nouns in train set
clf2.fit(train_data,labels_train)     #train on nouns in train set
clf3.fit(train_data,labels_train)     #train on nouns in train set
eclf.fit(train_data,labels_train)     #train on nouns in train set

pred1=clf1.predict(test_data)			#predict on nouns in test set
pred2=clf2.predict(test_data)			#predict on nouns in test set
pred3=clf3.predict(test_data)			#predict on nouns in test set
pred=eclf.predict(test_data)			#predict on nouns in test set

print "\nAccuracy of classifiers using nouns and average stars\n"
print "Multinomial_NB : ",accuracy_score(pred1,labels_test) #print accuracy
print "Logistic  : ",accuracy_score(pred2,labels_test) #print accuracy
print "Randomforest : ",accuracy_score(pred3,labels_test) #print accuracy
print "Voting : ",accuracy_score(pred,labels_test) #print accuracy

print "\nBest parameters",clf1.best_params_
print "Best parameters",clf2.best_params_
print "Best parameters",clf3.best_params_

#---------Run classifier based on nouns,average_stars and distance 
cols_with_loc = ['average_stars','dist']
train_data_loc = np.concatenate((counts_train.toarray(), train[cols_with_loc].as_matrix()), axis=1)
test_data_loc = np.concatenate((counts_test.toarray(), test[cols_with_loc].as_matrix()), axis=1)


clf1.fit(train_data_loc,labels_train)     #train on nouns in train set
clf2.fit(train_data_loc,labels_train)     #train on nouns in train set
clf3.fit(train_data_loc,labels_train)     #train on nouns in train set
eclf.fit(train_data_loc,labels_train)     #train on nouns in train set

pred1=clf1.predict(test_data_loc)			#predict on nouns in test set
pred2=clf2.predict(test_data_loc)			#predict on nouns in test set
pred3=clf3.predict(test_data_loc)			#predict on nouns in test set
pred=eclf.predict(test_data_loc)			#predict on nouns in test set

print "\nAccuracy of classifiers using  distance\n"
print "Multinomial_NB : ",accuracy_score(pred1,labels_test) #print accuracy
print "Logistic  : ",accuracy_score(pred2,labels_test) #print accuracy
print "Randomforest : ",accuracy_score(pred3,labels_test) #print accuracy
print "Voting : ",accuracy_score(pred,labels_test) #print accuracy

print "\nBest parameters",clf1.best_params_
print "Best parameters",clf2.best_params_
print "Best parameters\n",clf3.best_params_

