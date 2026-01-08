import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

'''
the first import  is to prevent the message from the lightgbm that is happening 
'''

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor

from src.logger import logging
from src.exceptions import CustomException

model_folder = os.getenv("MODEL_DIR", "models")
os.makedirs(model_folder, exist_ok=True)


class TRAINING_MODEL:

    def model_selection(self, df: pd.DataFrame):
      
        y = df['log_price']
        X = df.drop(columns=['log_price'])

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        try:

            models = {
                'ridge': {
                    'model': Ridge(),
                    'params': {
                        'model__alpha': [0.1, 1.0, 10.0]
                    }
                },
                'lasso': {
                    'model': Lasso(max_iter=5000),
                    'params': {
                        'model__alpha': [0.001, 0.01, 0.1]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'model__n_estimators': [200, 400],
                        'model__learning_rate': [0.05, 0.1],
                        'model__max_depth': [3, 5]
                    }
                },
                'xgboost': {
                    'model': XGBRegressor(
                        random_state=42,
                        verbosity=0,
                        objective='reg:squarederror'
                    ),
                    'params': {
                        'model__n_estimators': [300, 600],
                        'model__learning_rate': [0.05, 0.1],
                        'model__max_depth': [4, 6]
                    }
                }

                #,
                #'lightgbm': {
                #    'model': LGBMRegressor(
                #        random_state=42,
                #        verbosity=-1
                #    ),
                #    'params': {
                #        'model__n_estimators': [200, 500],
                #        'model__learning_rate': [0.05, 0.1]
                #    }
                #}
            }

            
            best_pipeline = None
            best_score = np.inf
            best_name = None

            for name, config in models.items():

                pipeline = Pipeline(steps=[
                    ('preprocess', preprocessor),
                    ('model', config['model'])
                ])

                grid = GridSearchCV(
                    pipeline,
                    config['params'],
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )

                grid.fit(X, y)

                rmse = -grid.best_score_

                print(f"{name} | RMSE: {rmse:.4f}")
                print(f"Best params: {grid.best_params_}\n")

                if rmse < best_score:
                    best_score = rmse
                    best_pipeline = grid.best_estimator_
                    best_name = name

            print(f"Best model: {best_name} (RMSE={best_score:.4f})")

        
            
            model_path = os.path.join(model_folder, "best_pipeline.pkl")
            joblib.dump(best_pipeline, model_path)

            print(f"Model saved to: {model_path}")
        
        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise

        return best_pipeline, best_score
