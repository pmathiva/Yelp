# Yelp challenge
The script performs sentiment analysis of reviews with relation to distance.It answers the challenge question 'Whether the reviews's behavior change with distance?'

Data: https://www.yelp.com/dataset_challenge

yelp_parser
------------
The script parses the data in JSON format to CSV files using the pandas library

* Execute the yelp_parser.py 

Set the location of the input file and the location where the parsed data sets have to be written

Output: 3 data sets namely, business.csv, review.csv and user.csv
 

User location and Business location
------------------------------------
* Execute the yelp_user_loc.py

   Set the location of the input files business.csv, review.csv and the location where user_loc.csv has to be written.
   
   Output: 1 data set user_loc.csv which gives the location of each user based on a clustering algorithm.
   
* Execute the yelp_business_loc.py

   Set the location of the input file business.csv and the location where business_loc.csv has to be written.
   
   Output: 1 data set business_loc.csv which gives the locaton of each business.
   
Data Cleansing and Integration
-------------------------------
* Execute the yelp_cleanser.py 

   Set the location of the input files review.csv,user.csv,business_loc.csv,user_loc.csv and the location where the cleansed data has to be written.
   
   Output: 1 cleansed data set namely, model_data.csv

Methodology
------------
Different classifiers are used and their performances are compared

- MultinomialNB

- Logistic

- Random forest

- Voting

* Classifiers are used with and without the distance as a feature 

* Other features – Nouns, average stars

* Execute the yelp_classifier.py

  Set the location of the input file: model_data.csv 
  
  Output: Accuracy of different classifiers


