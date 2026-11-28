"""Functions for evaluation in the splines and GAM walkthrough."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .additive import gam_model, gam_xy

# Step 14 - test_comparison

def test_rmse(models, X_test, y_test):
    results = {}

    # Calculate RMSE for each model
    for name, (model, setting) in models.items():
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results[name] = round(rmse, 2)

    return results


def gam_test_rmse(train, test):
    # Prepare training and testing data
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    # Fit GAM model and calculate test RMSE
    model = gam_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return round(rmse, 2)


def comparison_lines(rmse_by_model, gam_rmse):
    lines = []

    # Sort models by RMSE
    for name, (rmse, setting) in sorted(
        rmse_by_model.items(), key=lambda x: x[1][0]
    ):
        lines.append(f"{name:8s} rmse={rmse:6.2f} setting={setting}")

    lines.append(f"{'gam':8s} rmse={gam_rmse:6.2f} setting=age+year+education")

    return lines

__all__ = ['test_rmse', 'gam_test_rmse', 'comparison_lines']
