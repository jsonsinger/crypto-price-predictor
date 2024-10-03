from typing import Optional

import numpy as np
import optuna
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


class XGBoostModel:
    """
    XGBoost model with Optuna for hyperparameter tuning.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model: XGBRegressor = None #XGBRegressor(random_state=random_state)

    def fit(self,
            X_train,
            y_train,
            n_search_trials: Optional[int] = 0,
            n_splits: Optional[int] = 3):
        """
        Fit the model to the training data.
        """
        
        assert n_search_trials >= 0, "n_search_trials must not be negative"
        
        logger.info(f"Fitting model with {n_search_trials} search trials and {n_splits} splits")

        if n_search_trials == 0:
            self.model = XGBRegressor(random_state=self.random_state)
            self.model.fit(X_train, y_train)
        else:
            best_hyperparameters = self.__find_best_hyperparameters(X_train, y_train, n_splits, n_search_trials, self.random_state)

            # Best hyperparameters: {'n_estimators': 378, 'max_depth': 3, 'learning_rate': 0.010264994656322912, 'subsample': 0.9988794894605038, 'colsample_bytree': 0.796074873530645}
            #Best hyperparameters: {'n_estimators': 536, 'max_depth': 5, 'learning_rate': 0.05533063036289089, 'subsample': 0.6030547660011443, 'co
            logger.info(f"Best hyperparameters: {best_hyperparameters}")

            self.model = XGBRegressor(**best_hyperparameters, random_state=self.random_state)
            self.model.fit(X_train, y_train)
            
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def __find_best_hyperparameters(self, X_train, y_train, n_splits: int, n_search_trials: int, random_state: int) -> dict:

        def objective(trial: optuna.Trial) -> float:
                params = {
                    # "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    # "max_depth": trial.suggest_int("max_depth", 1, 10),
                    # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    
                     # Tree-based parameters
                    "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    
                    # Learning rate
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    
                    # Regularization
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    
                    # Other
                    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2)
                }

                # Split X_train into n_splits folds with time-series-split
                tscv = TimeSeriesSplit(n_splits=n_splits)
                mae_scores = []
                for train_index, validation_index in tscv.split(X_train):
                    X_train_split, X_val_split = X_train.iloc[train_index], X_train.iloc[validation_index]
                    y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[validation_index]

                    model = XGBRegressor(**params, random_state=random_state)
                    model.fit(X_train_split, y_train_split)

                    # Evaluate the model on the validation set
                    y_pred = model.predict(X_val_split)
                    mae = mean_absolute_error(y_val_split, y_pred)
                    mape = np.mean(np.abs((y_val_split - y_pred) / y_val_split)) * 100
                    logger.info(f"MAE: {mae}, MAPE: {mape} for split {len(mae_scores)}")
                    mae_scores.append(mae)
                return np.mean(mae_scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_search_trials)

        return study.best_params
    
    def get_model(self):
        return self.model

    
