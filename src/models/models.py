"""
Model classes for ARIMAX, LSTM, LightGBM, XGBoost and other forecasting models.
Provides a diverse set of time series forecasting models.
"""
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb


class ARIMAXModel:
    """ARIMAX model wrapper."""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data, exog=None):
        """Fit ARIMAX model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.model = SARIMAX(train_data, exog=exog, order=self.order)
        self.fitted_model = self.model.fit(disp=False)
        return self.fitted_model
    
    def predict(self, steps, exog=None):
        """Generate predictions."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet.")
        return self.fitted_model.forecast(steps=steps, exog=exog)
    
    def save_model(self, filepath):
        """Save model to file."""
        import joblib
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet.")
        joblib.dump({'order': self.order, 'fitted_model': self.fitted_model}, filepath)
    
    def load_model(self, filepath):
        """Load model from file."""
        import joblib
        data = joblib.load(filepath)
        self.order = data['order']
        self.fitted_model = data['fitted_model']
        return self.fitted_model


class LightGBMModel:
    """LightGBM model wrapper for time series forecasting."""
    
    def __init__(self, params=None, num_boost_round=200):
        """
        Initialize LightGBM model.
        
        Args:
            params: Dictionary of LightGBM parameters
            num_boost_round: Number of boosting iterations
        """
        self.params = params or {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'seed': 42
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=['train', 'valid'] if len(valid_sets) > 1 else ['train']
        )
        
        return self.model
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """Load model from file."""
        self.model = lgb.Booster(model_file=filepath)
        return self.model
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance as dictionary."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        importance_array = self.model.feature_importance(importance_type=importance_type)
        
        # Return as dictionary mapping feature names to importance
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance_array))
        else:
            return dict(zip([f'f{i}' for i in range(len(importance_array))], importance_array))


class XGBoostModel:
    """XGBoost model wrapper for time series forecasting."""
    
    def __init__(self, params=None, num_boost_round=200):
        """
        Initialize XGBoost model.
        
        Args:
            params: Dictionary of XGBoost parameters
            num_boost_round: Number of boosting iterations
        """
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'verbosity': 0
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            verbose_eval=False
        )
        
        return self.model
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """Load model from file."""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        return self.model
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.get_score(importance_type=importance_type)


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else None
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
    if mape is not None:
        metrics['MAPE'] = mape
    
    return metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models on the same test set.
    
    Args:
        models_dict: Dictionary of {model_name: model_instance}
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for model_name, model in models_dict.items():
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                raise AttributeError(f"Model {model_name} doesn't have predict method")
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred)
            results[model_name] = metrics
            
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    return results


def get_best_model(comparison_results, metric='RMSE'):
    """
    Get the best performing model based on a metric.
    
    Args:
        comparison_results: Results from compare_models()
        metric: Metric to use for comparison (default: 'RMSE')
        
    Returns:
        Tuple of (best_model_name, best_score)
    """
    valid_results = {name: res for name, res in comparison_results.items() 
                     if 'error' not in res and metric in res}
    
    if not valid_results:
        return None, None
    
    best_model = min(valid_results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1][metric]
