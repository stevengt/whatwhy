import pandas as pd
from keywords_extractor import KeywordsExtractor 

def get_df_with_keywords_from_csv(filename, nrows=None):
    df = pd.read_csv(filename, nrows=nrows)
    extractor = KeywordsExtractor(df)
    df_with_keywords = extractor.add_what_and_why_keywords_to_df()
    return df_with_keywords

tweets_csv_file = "/home/zach/git/whatwhy/Tweets/all_tweets_aggregated.csv"
df = get_df_with_keywords_from_csv(tweets_csv_file, nrows=50 )
print(df)
