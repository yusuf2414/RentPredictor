import os
import joblib
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import mean_squared_error, r2_score

#from src.logger import logging
#from src.exceptions import CustomException


class RentPricePredictor:
    def __init__(self,  model_path: str):
        try:
            self.model = joblib.load(model_path)
            logging.info("Model pipeline loaded successfully")
        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise
           

    def predict(self, testing_df: pd.DataFrame):
        """
        Predict rent prices (log scale) for new data
        """
        try:
            predictions = self.model.predict(testing_df)
            return predictions
        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise
        

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "log_price"):
        """
        Evaluate model if ground truth is available
        """
        try:
            y_true = test_df[target_col]
            X_test = test_df.drop(columns=[target_col])

            y_pred = self.model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            return {
                "rmse": rmse,
                "r2": r2
            }

        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise
        

if __name__ == "__main__":
    MODEL_PATH = os.path.join("models", "best_pipeline.pkl")
    TEST_DATA_PATH = os.path.join("data_folder", "testing_data.csv")

    predictor = RentPricePredictor(MODEL_PATH)

    df_test = pd.read_csv(TEST_DATA_PATH)

    metrics = predictor.evaluate(df_test)

    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

        
