## importing modules
# data manipulation
import numpy as np
import pandas as pd

# sentiment analysis modules
import preprocessor as p # filters the unwanted elements of the tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # sentiment analysis module
SIA = SentimentIntensityAnalyzer()

# opening tweets csv file
with open("tweets.csv") as fp:
    raw_tweets_data = pd.read_csv(fp,engine='python',error_bad_lines=False)

# changing string dates column of dataframe to datetime objects
raw_tweets_data['time'] = pd.to_datetime(raw_tweets_data['date'],errors='coerce')

# removing bad rows
raw_tweets_data.dropna(inplace=True)

# removing date column that has been replaced by time column
raw_tweets_data.drop('date',axis=1,inplace=True)

#setting index to date (needed so that the following operations do not lose the time component)
raw_tweets_data.set_index('time',inplace=True)

# getting polarity scores (negativity, neutrality, positivity, compound) of every tweet that has been cleaned (p.clean)
# WARNING: this step takes about 20-30 minutes (the sentiment analysis takes some time )
raw_tweets_data['polarity']= raw_tweets_data['content'].apply(lambda x: SIA.polarity_scores(p.clean(x)))

# creating new colomns (neg, neu, pos, comp) on dataframe
df_polarity = raw_tweets_data['polarity'].apply(pd.Series)

# getting time as a column (not index)
df_polarity.reset_index(inplace=True)

# exporting dataframe to csv
df_polarity.to_csv('sentiments.csv')
