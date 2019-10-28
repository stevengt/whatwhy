import argparse
from .text_processing import get_df_from_file
from .fs_client import FileSystemBatchSource, FileSystemBatchDestination
from .s3_client import S3BatchSource, S3BatchDestination
from .sqs_client import SQSBatchSource, SQSBatchDestination
from .preprocessing import BatchPreprocessor
from .wh_phrases import WHPhrasesBatchProcessor

def get_batch_source(source_type, source_name, delete_when_complete):
    if source_type == "fs":
        return FileSystemBatchSource(source_name, delete_when_complete=delete_when_complete)
    elif source_type == "s3":
        source_names = source_name.split("/")
        return S3BatchSource(bucket_name=source_names[0], folder_name=source_names[1], delete_when_complete=delete_when_complete)
    elif source_type == "sqs":
        return SQSBatchSource(source_name)
    else:
        raise AttributeError(f"Unsupported batch source type {source_type}.")

def get_batch_destination(dest_type, dest_name):
    if dest_type == "fs":
        return FileSystemBatchDestination(dest_name)
    elif dest_type == "s3":
        dest_names = dest_name.split("/")
        return S3BatchDestination(bucket_name=dest_names[0], folder_name=dest_names[1])
    elif dest_type == "sqs":
        return SQSBatchDestination(dest_name)
    else:
        raise AttributeError(f"Unsupported batch destination type {dest_type}.")

def get_batch_processor(batch_processor_type, batch_source, batch_dest):
    if batch_processor_type == "preprocessing":
        return BatchPreprocessor(batch_source, batch_dest)
    elif batch_processor_type == "wh-phrases":
        return WHPhrasesBatchProcessor(batch_source, batch_dest)
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
    arggroup.add_argument("--process", choices=["preprocessing", "wh-phrases"] )

    parser.add_argument("--source-type", choices=["fs", "s3", "sqs"], required=True)
    parser.add_argument("--source-name", required=True, help="If using S3, use the format bucket-name/folder-name.")
    parser.add_argument("--dest-type", choices=["fs", "s3", "sqs"], required=True)
    parser.add_argument("--dest-name", required=True, help="If using S3, use the format bucket-name/folder-name.")

    parser.add_argument("--delete-when-complete", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    batch_dest = get_batch_destination(args.dest_type, args.dest_name)
    batch_size = args.batch_size
    delete_when_complete = args.delete_when_complete

    if args.populate:
        df_file_name = args.source_name
        populate(df_file_name, batch_dest, batch_size)
    else:
        batch_source = get_batch_source(args.source_type, args.source_name, delete_when_complete)
        batch_processor_type = args.process
        batch_processor = get_batch_processor(batch_processor_type, batch_source, batch_dest)
        process(batch_processor)

if __name__ == "__main__":
    main()
