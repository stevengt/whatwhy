import pickle
import pandas as pd
import numpy as np
from whatwhy.webservices.wh_phrase_extractor.client import WHPhraseExtractorClient
from whatwhy.data_cleaner import remove_reply_tag_from_tweet_text, autocorrect_spelling_and_grammar

tweets_csv_file = "/home/stevengt/Documents/code/whatwhy/Tweets/all_tweets_aggregated.csv"
client = WHPhraseExtractorClient(request_timeout_in_seconds=180)
df = pd.read_csv(tweets_csv_file)
df_shuffled_index = np.random.permutation(df.index)
batch_size = 10

def get_next_batch(df, df_shuffled_index, batch_size, cur_batch_num):
    start_index = batch_size * cur_batch_num
    end_index = batch_size * (cur_batch_num + 1)
    batch_indeces = df_shuffled_index[start_index:end_index]
    return df.iloc[batch_indeces]

def save_batch_results(list_of_batch_results, batch_num):
    with open(f"/home/stevengt/Documents/code/whatwhy/Tweets/PROCESSED/5W1H/batch{batch_num}.pickle","wb") as out_file:
        pickle.dump(list_of_batch_results, out_file)

cur_batch_num = 0
while True:
    list_of_batch_results = []
    try:
        for i in range(1000):
            try:
                cur_batch = get_next_batch(df, df_shuffled_index, batch_size, cur_batch_num)
                cur_batch["Text"] = cur_batch["Text"].map( remove_reply_tag_from_tweet_text ) \
                                                     .map( autocorrect_spelling_and_grammar )
                cur_batch_results = client.get_wh_phrases_from_df(cur_batch, id_col_name="Tweet ID", content_col_name="Text")
                list_of_batch_results.append(cur_batch_results)
            except Exception as e:
                print(e)
            cur_batch_num += 1
        save_batch_results(list_of_batch_results, cur_batch_num)
    except Exception as e:
        print(e)
