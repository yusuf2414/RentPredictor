import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

'''
the first import is to prevent the message from the lightgbm that is happening
'''

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import ( RandomForestRegressor,GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

from src.logger import logging
from src.exceptions import CustomException


model_folder = os.getenv("MODEL_DIR", "models")
os.makedirs(model_folder, exist_ok=True)


class TRAINING_MODEL:

    def model_selection(self, df: pd.DataFrame):
        
        '''
        if you are doing this locally without EC2 Instance you dont need this 
        mlflow.set_tracking_uri("http://18.144.41.74:5000") always change the IP address
        accordingly
        '''

        mlflow.set_tracking_uri("http://18.144.41.74:5000")

        mlflow.set_experiment("Rent_Price_Prediction")
        '''
        Always create a container for this , because it helps with 
        one making sure that the 
        '''

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
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'model__n_estimators': [200, 500],
                        'model__max_depth': [None, 10, 20],
                        'model__min_samples_split': [2, 5],
                        'model__min_samples_leaf': [1, 2]
                    }
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'model__max_depth': [None, 5, 10, 20],
                        'model__min_samples_split': [2, 5, 10],
                        'model__min_samples_leaf': [1, 2, 4]
                    }
                }
            }

            best_pipeline = None
            best_score = np.inf
            best_name = None
            best_run_id = None

            logging.info("Training the data begins")

            for name, config in models.items():

                with mlflow.start_run(run_name=name):
                    '''
                    Docstring for tracking machine learning model
                     Metrics overite each other when we run so 
                     we need this in order to have traceability of each
                     machine learning model
                    '''

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

                    mlflow.log_param("model_name", name)
                    for param, value in grid.best_params_.items():

                        '''
                        Stores hyperparameters used for that run.
                        '''

                        mlflow.log_param(param, value)

                    '''
                    Tracks numeric performance metrics.
                    '''
                    mlflow.log_metric("rmse", rmse)

                    mlflow.sklearn.log_model(
                        grid.best_estimator_,
                        name="model"
                    )

                    logging.info(f"Training completed for model: {name}")

                    print(f"{name} | RMSE: {rmse:.4f}")
                    print(f"Best params: {grid.best_params_}\n")

                    if rmse < best_score:
                        best_score = rmse
                        best_pipeline = grid.best_estimator_
                        best_name = name
                        best_run_id = mlflow.active_run().info.run_id

            print(f"Best model: {best_name} (RMSE={best_score:.4f})")
            logging.info(f"Best Model selected: {best_name}")

            model_path = os.path.join(model_folder, "best_pipeline.pkl")
            joblib.dump(best_pipeline, model_path)

            '''
            local back up with Joblib that helps to easily get the model,
            if you dont wnat to use mlflow
            '''


            print(f"Model saved to: {model_path}")
            logging.info(f"Best model saved locally at {model_path}")

            mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model",
                name="RentPriceModel"
            )

        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

        return best_pipeline, best_score
