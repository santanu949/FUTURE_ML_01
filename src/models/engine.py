import optuna
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Any

class ModelEngine:
    """Core engine for model training, tuning, and comparison."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.best_params = {}
        self.best_model = None
        self.model_type = None

    def tune_prophet(self, n_trials=20):
        """Bayesian optimization for Prophet hyperparameters."""
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
            }
            m = Prophet(**params)
            m.fit(self.data)
            
            # Cross-validation
            df_cv = cross_validation(m, initial='100 days', period='30 days', horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv)
            return df_p['rmse'].values[0]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return study.best_params

    def train_best_prophet(self):
        """Trains Prophet with best tuned parameters."""
        m = Prophet(**self.best_params)
        m.fit(self.data)
        self.best_model = m
        self.model_type = 'prophet'
        return m

    def train_xgboost(self, test_size=30):
        """Trains an XGBoost model on lag features."""
        from ..data.processor import DataProcessor
        
        # Prepare data for ML
        df = self.data.copy()
        df = DataProcessor.feature_engineering(df.set_index('ds'), 'y').reset_index()
        
        train = df[:-test_size]
        test = df[-test_size:]
        
        features = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30', 'day_of_week', 'is_weekend']
        X_train, y_train = train[features], train['y']
        X_test, y_test = test[features], test['y']
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return model, rmse

    def compare_and_select(self):
        """Benchmarks Prophet vs XGBoost and auto-selects."""
        print("Benchmarking models...")
        
        # Tune and train Prophet
        self.tune_prophet(n_trials=5)
        prophet_m = self.train_best_prophet()
        
        # Train XGBoost
        xgb_m, xgb_rmse = self.train_xgboost()
        
        # Get Prophet RMSE from last CV
        df_cv = cross_validation(prophet_m, initial='100 days', period='30 days', horizon='30 days')
        prophet_rmse = performance_metrics(df_cv)['rmse'].mean()
        
        print(f"Prophet RMSE: {prophet_rmse}")
        print(f"XGBoost RMSE: {xgb_rmse}")
        
        if prophet_rmse < xgb_rmse:
            print("Selected Prophet as the best model.")
            self.best_model = prophet_m
            self.model_type = 'prophet'
        else:
            print("Selected XGBoost as the best model.")
            self.best_model = xgb_m
            self.model_type = 'xgboost'
            
        return self.best_model, self.model_type
