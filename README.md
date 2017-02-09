# Yelp
Instructions to execute the program

1) Datafile is from the yelp website  
   Filename: yelp_dataset_challenge_academic_dataset
2) Execute the yelp_parser.py 
   Set the location of the input file and the location where the parsed data sets have to be written.
   Output: 3 data sets namely, business.csv, review.csv and user.csv
3) Execute the yelp_user_loc.py
   Set the location of the input files business.csv, review.csv and the location where user_loc.csv has to be written.
   Output: 1 data set user_loc.csv which gives the location of each user based on a clustering algorithm.
4) Execute the yelp_business_loc.py
   Set the location of the input file business.csv and the location where business_loc.csv has to be written.
   Output: 1 data set business_loc.csv which gives the locaton of each business.
5) Execute the yelp_cleanser.py 
   Set the location of the input files review.csv,user.csv,business_loc.csv,user_loc.csv and the location where the cleansed data has to be written.
   Output: 1 cleansed data set namely, model_data.csv
6) Execute the yelp_classifier.py
   Set the location of the input file: model_data.csv 
   Output: Accuracy of different classifiers
