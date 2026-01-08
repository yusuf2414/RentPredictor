import os
import pandas as pd 
from src.data_sourcing import DataImportation
from model_trainer.training_model import TRAINING_MODEL
from predict_folder.predictor import RentPricePredictor


importer = DataImportation()
df = importer.datareading()
df = importer.transformation()
df=  importer.remove_columns()

#Model Training 
### intiation of the trained data 
training_model = TRAINING_MODEL()
best_pipeline, best_score = training_model.model_selection(df)


MODEL_PATH = os.path.join("models", "best_pipeline.pkl")
TEST_DATA_PATH = os.path.join("data_folder", "testing_data.csv")

predictor = RentPricePredictor(MODEL_PATH)

df_test = pd.read_csv(TEST_DATA_PATH)

metrics = predictor.evaluate(df_test)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")






