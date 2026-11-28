"""Functions for validation in the splines and GAM walkthrough."""

import numpy as np
from sklearn.model_selection import cross_val_score

# Step 3 - cv_tools


def cv_mse(model, X, y, cv):
    # Perform cross-validation and calculate negative MSE for each fold
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error"
    )

    # Convert negative MSE into positive MSE
    mse = -scores

    # Calculate average MSE across all folds
    mean = np.mean(mse)

    # Calculate standard error using sample standard deviation
    se = np.std(mse, ddof=1) / np.sqrt(len(mse))

    # Return mean and standard error rounded to 1 decimal
    return round(float(mean), 1), round(float(se), 1)


def cv_curve(make_model, X, y, values, cv):
    # Store mean MSE and standard error for each model
    means = []
    ses = []

    # Iterate through different model settings
    for value in values:

        # Create a model using the current setting
        model = make_model(value)

        # Evaluate the model using cross-validation
        mean, se = cv_mse(model, X, y, cv)

        # Store the evaluation results
        means.append(mean)
        ses.append(se)

    return means, ses


def one_se_rule(values, means, ses, prefer='smaller'):

    # Find the best mean
    best_idx = np.argmin(means)

    # Calculate one-SE threshold
    threshold = means[best_idx] + ses[best_idx]

    # Find eligible values
    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    # Choose preferred value
    if prefer == "smaller":
        return min(eligible)
    elif prefer == "larger":
        return max(eligible)
    else:
        raise ValueError("prefer must be 'smaller' or 'larger'")

__all__ = ['cv_mse', 'cv_curve', 'one_se_rule']
