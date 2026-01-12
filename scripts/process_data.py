import pandas as pd
import os
from src.s3_utils import download_from_s3, upload_to_s3
import logging

from src.data_sourcing import DataImportation


BUCKET = "rentpredictionyusuf"

RAW_KEY = "raw_data/apartments_for_rent_classified_100K.csv"
PROCESSED_KEY = "processed_data/training_data.csv"

LOCAL_RAW = "tmp_raw.csv"
LOCAL_PROCESSED = "tmp_processed.csv"

# 1. Download raw data
download_from_s3(BUCKET, RAW_KEY, LOCAL_RAW)

# 2. Load & process
importer = DataImportation()
df = importer.datareading(LOCAL_RAW)
logging.info(f"Data has been read")
df = importer.transformation()
logging.info(f"Data has been transformed")
df =  importer.remove_columns()
logging.info(f"Unneccessary columns removed from the data")

# 3. Save processed
df.to_csv(LOCAL_PROCESSED, index=False)

# 4. Upload processed data
upload_to_s3(
    local_path=LOCAL_PROCESSED,
    #BUCKET_NAME=BUCKET,
    s3_key=PROCESSED_KEY
)

print(f"Processed data uploaded → s3://{BUCKET}/{PROCESSED_KEY}")

'''
you can import classes that can help you obtain the pipeline but since i already have it
I will just processed and get it as shown below
from model_trainer.training_model import TRAINING_MODEL
training_model = TRAINING_MODEL()
best_pipeline, best_score = training_model.model_selection(df)

'''



LOCAL_MODEL_PATH = os.path.join("models", "best_pipeline.pkl")
MODEL_KEY = "models/best_pipeline.pkl"

upload_to_s3(
    local_path=LOCAL_MODEL_PATH,
    #bucket=BUCKET,
    s3_key=MODEL_KEY
)

print(f"Model uploaded → s3://{BUCKET}/{MODEL_KEY}")
