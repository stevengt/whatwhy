import ast
import numpy as np

def get_text_as_list(text):
    """Converts a string representation of a list to a list object."""
    if text is None or text is np.nan:
        return []
    else:
        return ast.literal_eval(text)

def get_default_token(word2vec_model):
    """
    Returns the token from a Word2Vec model to use as a default
    for end-of-sequence marking.
    """
    return "." if "." in word2vec_model.vocab.keys() else word2vec_model.index2word[0]

def get_token_counts(df, col_name):
    """
    Given a DataFrame column containing lists of tokens, this returns
    a dictionary of all tokens and the number of times they appear in
    the column.
    """
    counts = {}
    for tokens in df[col_name]:
        for token in tokens:
            count = counts.get(token)
            if count is None:
                count = 0
            counts[token] = count + 1
    return counts

def remove_uncommon_whatwhy_tokens(df, min_token_count):
    """
    Removes infrequently occurring tokens from the 'what tokens'
    and 'why tokens' columns in a DataFrame.
    """
    what_counts = get_token_counts(df, "what tokens")
    why_counts = get_token_counts(df, "why tokens")
    df["what tokens"] = df["what tokens"].apply(lambda x: [token for token in x if what_counts[token] >= min_token_count ])
    df["why tokens"] = df["why tokens"].apply(lambda x: [token for token in x if why_counts[token] >= min_token_count ])
    return df
