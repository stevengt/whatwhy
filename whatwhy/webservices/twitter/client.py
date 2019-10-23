import math
import time
import pandas as pd
import numpy as np
from twython import Twython


class TwitterClient():
    """Helper class to retrieve tweet texts given a list of tweet ID's using the Twitter API."""

    def __init__(self,
                 api_consumer_key=None,
                 api_consumer_secret=None,
                 api_access_token_key=None,
                 api_access_token_secret=None,
                 twython_api_instance=None):
        """
        Constructor that accepts either:
            - A Twython API instance.
            - Twitter API credentials.
        """
        if twython_api_instance is not None:
            self.api = twython_api_instance
        else:
            self.api = Twython( api_consumer_key, api_consumer_secret, api_access_token_key, api_access_token_secret )

    def get_empty_tweets_df(self):
        df = pd.DataFrame(columns=["Text"])
        df.index.name = "Tweet ID"
        return df

    def write_df_to_file(self, df, file_name):
        with open(file_name,"w") as new_file:
            df.to_csv(new_file)
    
    def get_tweets_batch_from_ids(self, id_batch):
        """Returns a list of tweets given a list of up to 100 tweet ID's."""
        id_batch_as_string = ""
        for id in id_batch:
            id_batch_as_string += id.replace("\n","") + ","
        return self.api.lookup_status( id=id_batch_as_string, tweet_mode="extended" )

    def get_tweet_texts_batch_from_ids(self, id_batch):
        """
        Returns a pandas DataFrame containing tweet ID's and their text content,
        given a list of up to 100 tweet ID's.
        """
        df = self.get_empty_tweets_df()
        tweets = self.get_tweets_batch_from_ids(id_batch)
        for tweet in tweets:
            full_text = ""
            if "retweeted_status" in tweet.keys():
                full_text = tweet["retweeted_status"]["full_text"]
            else:
                full_text = tweet["full_text"]
            df.loc[ tweet["id"] ] = full_text.replace("\n"," ").replace("\r"," ").encode('ascii', 'ignore').decode('ascii')
        return df

    def append_tweet_texts_batch_to_df_from_ids(self, df, id_batch):
        """Gets tweet texts from a batch of ID's and appends them to a pandas DataFrame."""
        df = df.append( self.get_tweet_texts_batch_from_ids(id_batch) )
        return df

    def get_all_tweet_texts_from_ids(self, ids=None, id_file_name=None, target_dir_name=None):
        """
        Returns a pandas DataFrame containing tweet ID's and their text content,
        given a list with any number of tweet ID's, or the name of a file containing
        a tweet ID on each line.
        Optionally, a directory name can be specified with target_dir_name to store
        batches of tweets in CSV files, instead of returning a pandas DataFrame
        (this will overwrite any previous CSV files).
        """

        if id_file_name is not None:
            with open(id_file_name, "r") as id_file:
                ids = id_file.readlines()

        df = self.get_empty_tweets_df()
        num_tweets = len(ids)
        max_num_tweets_per_api_call = 100

        for i in range( math.ceil( num_tweets / max_num_tweets_per_api_call ) ):
            print(f"{i*max_num_tweets_per_api_call}/{num_tweets} tweets downloaded.")            
            id_batch = ids[ i * max_num_tweets_per_api_call : np.min([ (i+1) * max_num_tweets_per_api_call, num_tweets ]) ] 
          
            while True: 
                try:
                    df = self.append_tweet_texts_batch_to_df_from_ids(df, id_batch)
                    break
                except:
                    # We have reached the maximum number of queries the Twitter API will
                    # allow in a given amount of time.
                    if target_dir_name is not None and df["Text"].count() > 0:
                        file_name = target_dir_name + f"/tweets{i}.csv"
                        self.write_df_to_file(df, file_name)
                        df = self.get_empty_tweets_df()
                    time.sleep(60) # Sleep until we can query the Twitter API again.
                    continue

        if target_dir_name is not None and df["Text"].count() > 0:
            file_name = target_dir_name + "/tweets.csv"
            self.write_df_to_file(df, file_name)
            return
        else:
            return df

