import os
import pandas as pd

from src.data_sourcing import DataImportation
from model_trainer.training_model import TRAINING_MODEL
from src.s3_utils import download_from_s3, upload_to_s3

BUCKET = "rentpredictionyusuf"

RAW_KEY = "raw_data/apartments_for_rent_classified_100K.csv"
PROCESSED_KEY = "processed_data/training_data.csv"
MODEL_KEY = "models/best_pipeline.pkl"

BASE_DIR = os.getcwd()

LOCAL_RAW = os.path.join(BASE_DIR, "tmp_raw.csv")
LOCAL_PROCESSED = os.path.join(BASE_DIR, "tmp_processed.csv")
LOCAL_MODEL = os.path.join(BASE_DIR, "models", "best_pipeline.pkl")

os.makedirs("models", exist_ok=True)

print("Downloading raw data from S3...")
download_from_s3(BUCKET, RAW_KEY, LOCAL_RAW)

print("Processing raw data...")
importer = DataImportation()
df = importer.datareading(LOCAL_RAW)
df = importer.transformation()
df = importer.remove_columns()

df.to_csv(LOCAL_PROCESSED, index=False)

print("Uploading processed training data to S3...")
upload_to_s3(
    local_path=LOCAL_PROCESSED,
    #bucket=BUCKET,
    s3_key=PROCESSED_KEY
)

print("Training model...")
training_model = TRAINING_MODEL()
best_pipeline, best_score = training_model.model_selection(df)

print(f"Best RMSE: {best_score:.4f}")

print("Uploading trained model to S3...")
upload_to_s3(
    local_path=LOCAL_MODEL,
    #bucket=BUCKET,
    s3_key=MODEL_KEY
)

print("Training pipeline completed successfully.")
print(f"Model saved to → s3://{BUCKET}/{MODEL_KEY}")
