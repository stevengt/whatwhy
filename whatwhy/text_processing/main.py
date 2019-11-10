import argparse
from .helper_methods import get_df_from_file
from .clients import FileSystemBatchSource, FileSystemBatchDestination, S3BatchSource, S3BatchDestination, SQSBatchSource, SQSBatchDestination
from .batch_processors import BatchTransferer, BatchPreprocessor, WHPhrasesBatchProcessor, BatchTokenizer, BatchWHPhrasesTokenizer, BatchConsolidator

def get_batch_source(source_type, source_name, delete_when_complete):
    if source_type == "fs":
        return FileSystemBatchSource(source_name, delete_when_complete=delete_when_complete)
    elif source_type == "s3":
        source_names = source_name.split("/")
        bucket_name = source_names[0]
        folder_name = "/".join(source_names[1:])
        return S3BatchSource(bucket_name=bucket_name, folder_name=folder_name, delete_when_complete=delete_when_complete)
    elif source_type == "sqs":
        return SQSBatchSource(source_name)
    else:
        raise AttributeError(f"Unsupported batch source type {source_type}.")

def get_batch_destination(dest_type, dest_name):
    if dest_type == "fs":
        return FileSystemBatchDestination(dest_name)
    elif dest_type == "s3":
        dest_names = dest_name.split("/")
        bucket_name = dest_names[0]
        folder_name = "/".join(dest_names[1:])
        return S3BatchDestination(bucket_name=bucket_name, folder_name=folder_name)
    elif dest_type == "sqs":
        return SQSBatchDestination(dest_name)
    else:
        raise AttributeError(f"Unsupported batch destination type {dest_type}.")

def get_batch_processor(batch_processor_type,
                            batch_source,
                            batch_dest,
                            id_col_name=None,
                            source_col_name=None,
                            dest_col_name=None,
                            include_cols=None):
    kwargs = {
        "source" : batch_source, 
        "dest" : batch_dest,
        "id_col_name" : id_col_name,
        "source_col_name" : source_col_name,
        "dest_col_name" : dest_col_name,
        "include_cols" : include_cols
    }
    
    if batch_processor_type == "preprocessing":
        return BatchPreprocessor(**kwargs)
    elif batch_processor_type == "wh-phrases":
        return WHPhrasesBatchProcessor(**kwargs)
    elif batch_processor_type == "transfer":
        return BatchTransferer(**kwargs)
    elif batch_processor_type == "tokenize":
        return BatchTokenizer(**kwargs)
    elif batch_processor_type == "tokenize-wh-phrases":
        return BatchWHPhrasesTokenizer(**kwargs)
    elif batch_processor_type == "consolidate":
        return BatchConsolidator(**kwargs)
    else:
        raise AttributeError(f"Unsupported batch processor type {batch_processor_type}.")

def populate(df_file_name, batch_dest, batch_size):
    df = get_df_from_file(df_file_name)
    batch_dest.populate_from_df(df, batch_size)

def process(batch_processor):
    batch_processor.run()

def main():
    parser = argparse.ArgumentParser()

    arggroup = parser.add_mutually_exclusive_group(required=True)
    arggroup.add_argument("--populate", action="store_true")
    arggroup.add_argument("--process", choices=["preprocessing", "wh-phrases", "transfer", "tokenize", "tokenize-wh-phrases", "consolidate"] )

    parser.add_argument("-st", "--source-type", choices=["fs", "s3", "sqs"], required=True)
    parser.add_argument("-sn", "--source-name", required=True, help="If using S3, use the format bucket-name/folder/name.")
    parser.add_argument("-dt", "--dest-type", choices=["fs", "s3", "sqs"], required=True)
    parser.add_argument("-dn", "--dest-name", required=True, help="If using S3, use the format bucket-name/folder/name.")

    parser.add_argument("-d", "--delete-when-complete", default=False, action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=1000)

    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--source-col", default="Preprocessed Text")
    parser.add_argument("--dest-col", default="Processed Text")
    parser.add_argument("--include-cols", nargs="*", default=None)

    args = parser.parse_args()

    batch_dest = get_batch_destination(args.dest_type, args.dest_name)

    if args.populate:
        df_file_name = args.source_name
        populate(df_file_name, batch_dest, args.batch_size)
    else:
        batch_source = get_batch_source(args.source_type, args.source_name, args.delete_when_complete)
        batch_processor = get_batch_processor(batch_processor_type=args.process,
                                                batch_source=batch_source,
                                                batch_dest=batch_dest,
                                                id_col_name=args.id_col,
                                                source_col_name=args.source_col,
                                                dest_col_name=args.dest_col,
                                                include_cols=args.include_cols)
        process(batch_processor)

if __name__ == "__main__":
    main()
