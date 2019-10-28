import sys
import logging
import pandas as pd
from whatwhy.text_processing import SQSBatchDestination, FileSystemBatchDestination

logger = logging.getLogger(__name__)

# def main(queue_name, df):
#     queue = SQSBatchDestination(queue_name)
#     logger.info(f"Populating data to SQS queue with name {queue_name}")
#     queue.populate_from_df(df)

def main(folder_name, df):
    folder = FileSystemBatchDestination(folder_name)
    logger.info(f"Populating data to local folder {folder_name}")
    folder.populate_from_df(df)

if __name__ == "__main__":
    # queue_name = sys.argv[1]
    folder_name = sys.argv[1]
    csv_file_name = sys.argv[2]
    nrows = sys.argv[3]
    logger.info(f"Loading csv file at {csv_file_name}")
    df = pd.read_csv(csv_file_name, nrows)
    main(folder_name, df)
