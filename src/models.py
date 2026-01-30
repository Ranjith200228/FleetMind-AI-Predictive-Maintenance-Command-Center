from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float


def evaluate(y_true, y_pred, name: str) -> ModelResult:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return ModelResult(name=name, mae=mae, rmse=rmse)


def train_ridge(X_train, y_train):
    """
    Ridge baseline with:
    - median imputation (fixes NaNs from rolling std at early cycles)
    - scaling (Ridge benefits from standardized features)
    """
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Stronger model (non-linear) with:
    - median imputation (RF also needs no NaNs)
    """
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    return model
