# What, why?

## Installation

*WhatWhy* requires `swig3` to be installed (usually it is in your distro package manager).
If using the included Docker files to extract WH phrases from text 
(i.e., who, what, when, where, why, how), then `Docker` and `Docker Compose`
are also required.

To install, simply clone the repository and install with `pip`:
```
git clone https://github.com/stevengt/whatwhy.git
cd whatwhy
pip install .
```

There may be temporary build errors from the external dependency `jamspell`, but these can be safely ignored.

## Usage

To prepare a CSV data set of text for use during model training, use the `whatwhy-text` CLI.

```
usage: whatwhy-text [-h]
                    (--populate | --process {preprocessing,wh-phrases,transfer,tokenize,tokenize-wh-phrases,consolidate})
                    -st {fs,s3,sqs} -sn SOURCE_NAME -dt {fs,s3,sqs} -dn
                    DEST_NAME [-d] [-bs BATCH_SIZE] [--aws-region AWS_REGION]
                    [--id-col ID_COL] [--source-col SOURCE_COL]
                    [--dest-col DEST_COL]
                    [--include-cols [INCLUDE_COLS [INCLUDE_COLS ...]]]

This is a CLI for batch processing text data. Specifically, it is used
to preprocess text, extract WH phrases (who, what, when, where, why, how),
and prepare the extracted phrases as text tokens for further analysis
with machine learning tools.

Data is assumed to be in CSV format. To split a single CSV
file into multiple batch files, use the --populate argument. 
Similarly, to consolidate batch files into a single CSV file,
use the argument '--process consolidate'.    

Data must be read from, and written to, one of these supported locations:
    - Local File System
    - Amazon S3
    - Amazon SQS

If using AWS, credentials should be stored in a format compatible with boto3,
such as environment variables or a credentials file. For more information, see:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html

Supported batch processing tasks are:
    - consolidate   : Consolidates data from multiple CSV files into a single CSV file.
    - transfer      : Transfers a batch of data without changing its contents.
    - preprocessing : Preprocesses text by removing URL's and auto-correcting common spelling errors.
    - wh-phrases    : Extracts the WH phrases (who, what, when, where, why, how) from text.
                      This is intended to be run from within a Docker network, since access to
                      a Stanford CoreNLP server API at http://corenlp-service:9000 is required.
                      Please see the readme file at https://github.com/stevengt/whatwhy
                      for more information.
    - tokenize      : Tokenizes and standardizes text-segments.
    - tokenize-wh-phrases : This is identical to 'tokenize', except it tokenizes the 
                            columns 'who', 'what', 'when', 'where', 'why', 'how'
                            and stores the results in 'who tokens', 'what tokens', 
                            'when tokens', 'where tokens', 'why tokens', and 'how tokens'.

optional arguments:
  -h, --help            show this help message and exit
  --populate            Use this argument to split a single CSV file into multiple batch files.
  --process {preprocessing,wh-phrases,transfer,tokenize,tokenize-wh-phrases,consolidate}
  -st {fs,s3,sqs}, --source-type {fs,s3,sqs}
  -sn SOURCE_NAME, --source-name SOURCE_NAME
                        If using S3, use the format bucket-name/folder/name.
  -dt {fs,s3,sqs}, --dest-type {fs,s3,sqs}
  -dn DEST_NAME, --dest-name DEST_NAME
                        If using S3, use the format bucket-name/folder/name.
  -d, --delete-when-complete
                        Optional flag to delete batches from the source after processing them.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        The number of rows each CSV batch file should have if 
                        using the --populate flag.
  --aws-region AWS_REGION
                        Name of AWS region, if using SQS.
  --id-col ID_COL       Name of the column to treat as an index containing 
                        unique identifiers for data rows.
  --source-col SOURCE_COL
                        Name of the column to perform processing tasks on.
  --dest-col DEST_COL   Name of the column to store results in after processing data.
  --include-cols [INCLUDE_COLS [INCLUDE_COLS ...]]
                        By default, only the ID and destination columns will be written to 
                        the destination. Use this argument to specify any additional columns 
                        to include.
```

To extract the WH phrases (who, what, when, where, why, how) from text,
configure the following environment variables and run `make && make run-wh-phrase-extractor`
from the project root directory to build and run the included Docker files:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `WHATWHY_SOURCE_TYPE`
- `WHATWHY_SOURCE_NAME`
- `WHATWHY_DEST_TYPE`
- `WHATWHY_DEST_NAME`
- `WHATWHY_ID_COL_NAME`
- `WHATWHY_SOURCE_COL_NAME`

To use the resulting set of prepared data to train and use a model, use the `whatwhy-model` CLI.

```
usage: whatwhy-model [-h]
                     (--train | --predict PREDICT [PREDICT ...] | --compare-test | --compare-train)
                     [-csv CSV_FILE_NAME]
                     [--min-token-frequency MIN_TOKEN_FREQUENCY]
                     [-min-tokens MIN_TOKENS_PER_SAMPLE]
                     [-max-tokens MAX_TOKENS_PER_SAMPLE] [-bs BATCH_SIZE]
                     [--epochs EPOCHS]

This is a CLI for training and using a model to predict sequences of 'why' text from input 'what' text.

This script uses the GoogleNews gensim Word2Vec model to embed
text tokens into 300-dimensional vectors. The first time the
script is run it may take some time to download this model.

Also note that this model is quite large (about 3.6 GB), and any
models created by this script can also potentially be large
depending on the size of the provided data set. If you are using
a Linux system with a small root partition, then you should install
this script in a Python distribution (e.g., Anaconda) that stores
files on a large enough disk.

optional arguments:
  -h, --help            show this help message and exit
  --train               Trains a prediction model using a supplied CSV file or previously loaded 
                        data set. This will overwrite any previously trained models.
  --predict PREDICT [PREDICT ...]
                        Uses a previously trained model to predict a sequence of 'why' text 
                        from the input 'what' text.
  --compare-test        Uses a previously trained model to compare its predictions 
                        against its testing data set.
  --compare-train       Uses a previously trained model to compare its predictions 
                        against its training data set.
  -csv CSV_FILE_NAME, --csv-file-name CSV_FILE_NAME
                        Name of a tab delimited local CSV file containing a data set for model training. 
                        If left blank, the most recently loaded data set will be used. 
                        CSV files must include columns labeled 'what tokens' and 'why tokens', 
                        each containing plain-text representations of a Python list of strings.
  --min-token-frequency MIN_TOKEN_FREQUENCY
                        The minimum number of times a token should occur in the dataset 
                        to be used for training a WhatWhyPredictor model.
  -min-tokens MIN_TOKENS_PER_SAMPLE, --min-tokens-per-sample MIN_TOKENS_PER_SAMPLE
                        The minimum number of tokens a sample should contain to be used 
                        for training a WhatWhyPredictor model.
  -max-tokens MAX_TOKENS_PER_SAMPLE, --max-tokens-per-sample MAX_TOKENS_PER_SAMPLE
                        The maximum number of tokens a sample should contain for training 
                        a WhatWhyPredictor model. Any extra tokens will be truncated.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs EPOCHS
```

### Example Usage

