
import os
import pandas as pd
import pytest
from whatwhy.text_processing.helper_methods import get_df_from_csv_string
from whatwhy.text_processing.batch_processors.preprocessing import BatchPreprocessor

@pytest.fixture
def TEXT_CSV_DATA():
    with open( os.path.abspath( os.path.join(__file__, "../../../test_resources/text.csv") ), "r" ) as in_file:
        return in_file.read()

def test_get_batch_results(TEXT_CSV_DATA):
    batch_preprocessor = BatchPreprocessor(source=None, dest=None, dest_col_name="Actual Preprocessed Text", include_cols=["Preprocessed Text"])
    results = batch_preprocessor.get_batch_results(TEXT_CSV_DATA)
    df = get_df_from_csv_string(results["file_content"])
    num_rows = df.shape[0]
    for i in range(num_rows):
        row = df.iloc[i]
        expected = row["Preprocessed Text"]
        actual = row["Actual Preprocessed Text"]
        assert ( pd.isnull(expected) and pd.isnull(actual) ) or ( expected == actual )

def test_remove_url():
    batch_preprocessor = BatchPreprocessor(source=None, dest=None)
    assert batch_preprocessor.remove_url(None) is None
    assert batch_preprocessor.remove_url("lorem ipsum www.google.com") == "lorem ipsum "
    assert batch_preprocessor.remove_url("lorem ipsum") == "lorem ipsum"

def test_autocorrect_spelling():
    batch_preprocessor = BatchPreprocessor(source=None, dest=None)
    assert batch_preprocessor.autocorrect_spelling(None) is None
    assert batch_preprocessor.autocorrect_spelling("Tgis word is misspelled.") == "This word is misspelled."
    assert batch_preprocessor.autocorrect_spelling("This word is not misspelled.") == "This word is not misspelled."
