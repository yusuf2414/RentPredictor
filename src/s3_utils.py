import boto3
import os

s3 = boto3.client("s3")

BUCKET_NAME = "rentpredictionyusuf"


def upload_to_s3(local_path: str, s3_key: str):
    """
    Upload a local file to S3.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"{local_path} not found")

    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"Uploaded → s3://{BUCKET_NAME}/{s3_key}")

def download_from_s3(bucket, s3_key, local_path):
    s3.download_file(bucket, s3_key, local_path)


def upload_to_s3(local_path: str, s3_key: str):
    """
    Upload a local file to S3.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"{local_path} not found")

    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"Uploaded → s3://{BUCKET_NAME}/{s3_key}")