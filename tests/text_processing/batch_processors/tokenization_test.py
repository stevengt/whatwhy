import os
import pandas as pd
import pytest
from whatwhy.text_processing.helper_methods import get_df_from_csv_string
from whatwhy.text_processing.batch_processors.tokenization import BatchTokenizer

@pytest.fixture
def TEXT_CSV_DATA():
    with open( os.path.abspath( os.path.join(__file__, "../../../test_resources/text.csv") ), "r" ) as in_file:
        return in_file.read()

def test_get_batch_results(TEXT_CSV_DATA):
    batch_tokenizer = BatchTokenizer(source=None, dest=None, dest_col_name="Actual Tokens", include_cols=["Tokens"])
    results = batch_tokenizer.get_batch_results(TEXT_CSV_DATA)
    df = get_df_from_csv_string(results["file_content"])
    num_rows = df.shape[0]
    for i in range(num_rows):
        row = df.iloc[i]
        expected = row["Tokens"]
        actual = row["Actual Tokens"]
        assert ( pd.isnull(expected) and actual == "[]" ) or ( expected == actual )

def test_get_tokenized_column(TEXT_CSV_DATA):
    batch_tokenizer = BatchTokenizer(source=None, dest=None, dest_col_name="Actual Tokens", include_cols=["Tokens"])
    df = get_df_from_csv_string(TEXT_CSV_DATA)
    df["Actual Tokens"] = batch_tokenizer.get_tokenized_column(df, "Preprocessed Text")
    num_rows = df.shape[0]
    for i in range(num_rows):
        row = df.iloc[i]
        expected = row["Tokens"]
        actual = row["Actual Tokens"]
        assert ( pd.isnull(expected) and str(actual) == "[]" ) or ( expected == str(actual) )

def test_get_list_of_lemmatized_words_from_text():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.get_list_of_lemmatized_words_from_text(None)) == 0
    assert len(batch_tokenizer.get_list_of_lemmatized_words_from_text("NOT PROCESSED")) == 0
    text = "hello, this contains many words"
    expected_list = ["hello", "thi", "contain", "many", "word"]
    actual_list = batch_tokenizer.get_list_of_lemmatized_words_from_text(text)
    assert len(expected_list) == len(actual_list)
    for i in range(len(expected_list)):
        assert expected_list[i] == actual_list[i]

def test_convert_to_lowercase():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.convert_to_lowercase([])) == 0
    assert " ".join( batch_tokenizer.convert_to_lowercase(["HeLLo", "wORLd"]) ) == "hello world"

def test_remove_punctuation():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.remove_punctuation([])) == 0
    assert " ".join( batch_tokenizer.remove_punctuation(["shouldn't", "contain", "punct-ua_ti:on"]) ) == "shouldnt contain punctuation"

def test_remove_non_alphabetic_tokens():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.remove_non_alphabetic_tokens([])) == 0
    assert " ".join( batch_tokenizer.remove_non_alphabetic_tokens(["abc", ",", "easy", "as", "123"]) ) == "abc easy as"

def test_remove_stop_words():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.remove_stop_words([])) == 0
    assert " ".join( batch_tokenizer.remove_stop_words(["but", "please", "stop", "all", "the", "bickering"]) ) == "please stop bickering"

def test_remove_short_tokens():
    batch_tokenizer = BatchTokenizer(source=None, dest=None)
    assert len(batch_tokenizer.remove_short_tokens([])) == 0
    assert " ".join( batch_tokenizer.remove_short_tokens(["i", "want", "a", "cup", "of", "tea"]) ) == "want cup tea"

