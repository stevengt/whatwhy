import logging
import pandas as pd
from keywords_extractor import KeywordsExtractor 

logging.basicConfig(level=logging.INFO)

def get_df_with_keywords_from_csv(filename, nrows=None):
    logging.info("Loading CSV file...")
    df = pd.read_csv(filename, nrows=nrows)
    extractor = KeywordsExtractor(df)
    df_with_keywords = extractor.add_5w1h_keywords_to_df()
    return df_with_keywords

#tweets_csv_file = "/home/zach/git/whatwhy/Tweets/all_tweets_aggregated.csv"
tweets_csv_file = "/home/stevengt/Documents/code/whatwhy/Tweets/all_tweets_aggregated.csv"
df = get_df_with_keywords_from_csv(tweets_csv_file, nrows=50 )
# print(df)
