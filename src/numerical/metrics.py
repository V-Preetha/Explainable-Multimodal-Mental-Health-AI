from __future__ import annotations

import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, targets):
    result = {}
    for index, target in enumerate(targets):
        mse = mean_squared_error(y_true[:, index], y_pred[:, index])
        result[target] = {"mae": mean_absolute_error(y_true[:, index], y_pred[:, index]), "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2_score(y_true[:, index], y_pred[:, index]), "explained_variance": explained_variance_score(y_true[:, index], y_pred[:, index])}
    return result


