# module to get current directory folder
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# data manipulation
import numpy as np
import pandas as pd

# scraping modules
import snscrape.modules.twitter as twitter
import itertools

# date and time manipulation
import datetime
import pytz
tz = pytz.timezone("Etc/GMT")


# module for API
import requests
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json


# sentiment analysis modules
import preprocessor as p # filters the unwanted elements of the tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # sentiment analysis module
SIA = SentimentIntensityAnalyzer()

# technical (financial) indicators 
import ta  

# machine learning modules
import torch
import torch.nn as nn

# setting model as class (standard way of doing it)
class LSTMClassifier(nn.Module):
    # init method
    def __init__(self, input_size, hidden_size,sequence_size,num_layers,dropout):
        
        # calling super constructor
        super(LSTMClassifier,self).__init__()

        # initializing all classes variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.num_layers=num_layers
        self.droput = dropout
        
        # creating the LSTM cell of specified dimensions and characteristics
        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=num_layers,batch_first=True,dropout=0.2)

        # dimension reduction: we want the hidden state to be reducted to single output 
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

        # sigmoid function so that output is between 0 and 1 (classification)
        self.sigmoid = nn.Sigmoid()

    # forward method to set 
    def forward(self, x):
        
        # applying input of LSTM cells to get output and hidden states
        lstm_out, self.hidden = self.lstm(x)
        
        # getting output between 0 and 1 from last hidden state output of LSTM netword
        y_pred = self.sigmoid(self.linear(lstm_out[:,-1,:]))

        return y_pred

def launchPriceAPI():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {'start':'1','limit':'50','convert':'USD'}
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': '0ee334e0-37a8-406f-9f8c-93543ff21e8e'}

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        new_time = data['data'][0]['quote']['USD']['last_updated']
        new_price = data['data'][0]['quote']['USD']['price']
        new_volume = data['data'][0]['quote']['USD']['volume_24h']
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        new_time = np.nan
        new_price = np.nan
        new_volume = np.nan
    return new_time, new_price, new_volume

def tweets2polarity(tweets_list):
    neg_global = 0
    pos_global = 0
    if len(tweets_list)>0:
        for tweet in tweets_list:
            polarity = list(SIA.polarity_scores(p.clean(tweet)).values())
            neg_global += polarity[0]
            pos_global += polarity[1]
        return [neg_global/len(tweets_list),pos_global/len(tweets_list)]
    else: 
        return [0, 0]

def time_rounder(date_time_str, time_int):
    date_time = datetime.datetime.strptime(date_time_str, "%Y-%m-%dT%H:%M:%S.000Z")
    number = int(time_int[:-1])
    unit = time_int[-1]
    if unit == 'T':
        rounded_date_time = date_time - datetime.timedelta(minutes=date_time.minute % number,seconds=date_time.second,microseconds=date_time.microsecond)
    elif unit == 'H':
        rounded_data_time = date_time - datetime.timedelta(hours=date_time.hours % number, minutes=date_time.minute,seconds=date_time.second,microseconds=date_time.microsecond)
    elif unit == 'D':
        rounded_date_time = date_time - datetime.timedelta(days=date_time.days % number, hours=date_time.hours, minutes=date_time.minute,seconds=date_time.second,microseconds=date_time.microsecond)
        
    return rounded_date_time

def newRow(new_time, new_price, new_volume, time_int):
    # getting the date of today in string
    today = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d") 

    #creating generator for scraping
    tweets = twitter.TwitterSearchScraper(f"bitcoin since:{today} until: filter:has_engagement lang:en").get_items()

    # iterating through all tweets
    tweets = itertools.islice(tweets, None)

    # storing tweets in pandas dataframe
    raw_df = pd.DataFrame(tweets)

    # grouping by the time interval
    grouped_df = raw_df.groupby(pd.Grouper(key='date',freq=f"{time_int}"))

    # only keeping last row (last time interval that will be updated in the sequence)
    df = pd.DataFrame(grouped_df.content.apply(list)).tail(1)

    # getting polarity scores for all tweets in time interval
    df[['neg','pos']]=df.content.apply(tweets2polarity).to_list()

    # dropping tweets column
    df.drop('content',axis=1,inplace=True)

    # getting number of tweets in time interval
    df['tweets'] = grouped_df.content.count()

    # appending closing and volume to time interval
    df[['close','volume']] = pd.DataFrame({'close': new_price,'volume': new_volume}, index=[tz.localize(time_rounder(new_time,time_int))])

    df['rsi'] = np.nan
    return df

def prediction(prediction, threashold):
    if abs(prediction-0.5)>=threashold:
        if prediction>0.5:
            return 1
        else:
            return -1
    else:
        return 0
def main():
    
    # getting global parameters
    with open(os.path.join(sys.path[0],'parameters.json')) as fp:
        parameters = json.load(fp)
        
    TIME_INT = parameters['TIME_INT']
    RSI_WINDOW = int(parameters['RSI_WINDOW'])
    
    # getting dataframe 
    old_df = pd.read_parquet(os.path.join(sys.path[0],'data/dataframe.parquet'))
    
    # removing first row to get dataframe of sequence_size
    df = old_df.iloc[1:]
    
    # getting new price and volume of bitcoin
    new_time, new_price, new_volume = launchPriceAPI()
    
    # getting new row for sequence with new price and volume and sentiment analysis
    new_df = newRow(new_time=new_time, new_price=new_price, new_volume=new_volume,time_int=TIME_INT)
    
    # appending new row to dataframe
    df = df.append(new_df,ignore_index=True)

    # calculating rsi index
    df['rsi'] = ta.momentum.RSIIndicator(close=df.close, window=RSI_WINDOW).rsi()
    
    # converting missing data (in format: NaN): 
        # negativity and positivity to 0
        # closing, volume and rsi to last values 
    df.fillna({'neg':0,'pos':0},inplace=True)
    df.close.ffill(inplace=True)
    df.volume.ffill(inplace=True)
    df.rsi.ffill(inplace=True)
    
    # saving updated dataframe
    df.to_parquet(os.path.join(sys.path[0],'data/dataframe.parquet'))
    
    # removing first window_size columns as RSI needs at least window_size number of columns before computing (before: NA)
    final_df = df.iloc[20:]
    
    # converting dataframe to torch tensor
    sequence = torch.Tensor(final_df.to_numpy()).unsqueeze(dim=1)
    
    # loading ML model
    model = torch.load(os.path.join(sys.path[0],'pytorch_models/pretrained_model.pt'))
    
    # predicting value
    y_pred = model(sequence)
    
    # formatting the prediction, time interval and prediction (with threashold) to dataframe for exporting
    out = pd.DataFrame([y_pred[0].detach().numpy(),prediction(y_pred[0].detach().numpy(),0.1),TIME_INT], columns=[['prediction','threasholded_prediction','time_interval']] ,index=[datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d,%Hh%M")])

    # exporting csv file
    out.to_csv(os.path.join(sys.path[0],'data/live_preds.csv'),mode='a',header=False)

if __name__ == "__main__":
    main()

