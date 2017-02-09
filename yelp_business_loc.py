
import pandas as pd

df = pd.read_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\business.csv")
df['loc'] = zip(df['latitude'], df['longitude'])
df  = df.drop_duplicates(subset = "business_id")
print df.shape
print df.dtypes
df.to_csv("C:\\Users\\priya.cse2009\\Documents\\Python Scripts\\yelp\\BIA_660_Finalproject\\business_loc.csv", index = False)
