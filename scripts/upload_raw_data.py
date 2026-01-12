#from src.s3_utils import upload_to_s3

#upload_to_s3(
    #"src/apartments_for_rent_classified_100K.csv",
    #"raw_data/apartments_for_rent_classified_100K.csv"
#)



import os
from src.s3_utils import upload_to_s3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

local_file = os.path.join(
    PROJECT_ROOT,
    "src",
    "apartments_for_rent_classified_100K.csv"
)

upload_to_s3(
    local_path=local_file,
    bucket="rentpredictionyusuf",
    s3_key="raw_data/apartments_for_rent_classified_100K.csv"
)
