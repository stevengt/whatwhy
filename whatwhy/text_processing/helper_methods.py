from io import StringIO
import csv
import pandas as pd

def get_csv_string_from_df(df):
    with StringIO() as csv_stream:
        df.to_csv(csv_stream, sep="\t", quoting=csv.QUOTE_ALL, quotechar='"')
        return csv_stream.getvalue()

def get_df_from_csv_string(csv_string):
    with StringIO(csv_string) as csv_stream:
        return pd.read_csv(csv_stream, index_col=False, sep="\t", dtype=str, quoting=csv.QUOTE_ALL, quotechar='"')

def get_df_from_file(file_name):
    return pd.read_csv(file_name, index_col=False, sep="\t", dtype=str, quoting=csv.QUOTE_ALL, quotechar='"')
