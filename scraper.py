# importing necessary modules

import pandas as pd # data manipulation
from time import time  # 

import snscrape.modules.twitter as twitter # scraping module

import itertools
import datetime
import sys

# 
if len(sys.argv) != 4:
    print("Missing arguments or too many given")
    print(f"python main3.py [iterations] [tweets/day] [date]")
    exit()

else:

    max_iter = int(sys.argv[1])

    if sys.argv[2] == "None":
        n_iter = None
    else:
        n_iter = int(sys.argv[2])
    
    init_date = sys.argv[3]

# function that stores all tweets containing bitcoin during a day in a pandas dataframe
def scrape(date):
	# creating generator for scraping
    tweets = twitter.TwitterSearchScraper(f"bitcoin since:{yest2str(date)} until:{date} filter:has_engagement lang:en").get_items()
    # iterating through all tweets
    tweets = itertools.islice(tweets, n_iter)
    # storing tweets in pandas dataframe
    df = pd.DataFrame(tweets)
    # returning necessay columns of dataframe
    return df[['date','content']]

# lambda function to change date to the date of the day before
yest2str = lambda date: datetime.datetime.strftime(datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=-1), "%Y-%m-%d") 


def main():
	# initializing to initial date
    date = init_date
    # going through max_iter number of days
    for i in range(max_iter):
        print(f"iter: {i+1}/{max_iter}")
        s = time()
    	# scraping for date    
        df = scrape(date)
        e = time()
        print("Done")
        # creating dataframe that stores the date that has been scraped, the number of iterations, the time that the scraping occured and the duration of the scraping
        dft = pd.DataFrame({"date": yest2str(date), "iter": f"{i+1}/{max_iter}", "time": datetime.datetime.strftime(datetime.datetime.now(), "%d-%m %H:%M:%S"), "duration": [e-s]})
        # appending infos dataframe to csv file 
        dft.to_csv("history.csv", mode="a", index=False, header=False)
       	# appending tweets dataframe to csv file 
        df.to_csv("tweets.csv", mode="a", index=False, header=False)
    	# changing date to the date before
        date = yest2str(date)

# protocol to lauch script
if __name__ == "__main__":
    main()