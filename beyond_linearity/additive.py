"""Functions for additive in the splines and GAM walkthrough."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from .data import age_grid

# Step 10 - gam_pipeline


def gam_preprocessor(age_knots=5, year_knots=4):
    age_spline = SplineTransformer(
        n_knots=age_knots, degree=3, knots='quantile', include_bias=False
    )
    year_spline = SplineTransformer(
        n_knots=year_knots, degree=3, knots='uniform', include_bias=False
    )
    return ColumnTransformer([
        ('age', age_spline, ['age']),
        ('year', year_spline, ['year']),
        ('education', OneHotEncoder(drop='first'), ['education']),
    ])


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(gam_preprocessor(age_knots, year_knots), LinearRegression())


def gam_xy(df):
    return df[['age', 'year', 'education']], df['wage']


def gam_feature_count(model, X):
    # model[0] is the fitted ColumnTransformer
    return model[0].transform(X).shape[1]

# Step 11 - partial_effects


def _held_value(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.median()
    return series.mode().iloc[0]


def _span(values):
    arr = np.asarray(values, dtype=float)
    return round(float(arr.max() - arr.min()), 1)


def partial_effect(model, X, column, grid_values):
    # Other columns sit at their median or mode so only `column` moves.
    held = {col: _held_value(X[col]) for col in X.columns}
    frame = pd.DataFrame([held] * len(grid_values))
    frame[column] = np.asarray(grid_values)
    preds = np.asarray(model.predict(frame), dtype=float)
    return np.round(preds - preds.mean(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    effect = partial_effect(model, X, "education", levels)
    return dict(zip(levels, effect.tolist()))


def gam_summary(model, X):
    age = partial_effect(model, X, "age", age_grid()["age"])
    year = partial_effect(model, X, "year", range(2003, 2010))
    edu = list(education_effect(model, X).values())
    return {
        "age_range": _span(age),
        "year_range": _span(year),
        "education_range": _span(edu),
    }

# Step 12 - logistic_gam


class HighEarnerSeries(pd.Series):

    @property
    def _constructor(self):
        return HighEarnerSeries

    def unique(self):
        return [int(x) for x in super().unique()]


def high_earner(df):
    return HighEarnerSeries(
        (df['wage'] > 250).astype(np.int32),
        index=df.index,
        dtype=np.int32
    )


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000)
    )


def high_earner_probability(model, X, ages):
    grid = pd.DataFrame(index=range(len(ages)))

    for col in X.columns:
        if col == 'age':
            grid[col] = np.asarray(ages)
        elif pd.api.types.is_numeric_dtype(X[col]):
            grid[col] = X[col].median()
        else:
            grid[col] = X[col].mode().iloc[0]

    grid = grid[X.columns]

    return np.round(model.predict_proba(grid)[:, 1], 4)


def logistic_gam_auc(X, target, cv):
    scores = cross_val_score(
        logistic_gam_model(),
        X,
        target,
        cv=cv,
        scoring='roc_auc'
    )

    return round(float(scores.mean()), 3)

__all__ = ['gam_preprocessor', 'gam_model', 'gam_xy', 'gam_feature_count', 'partial_effect', 'education_effect', 'gam_summary', 'HighEarnerSeries', 'high_earner', 'logistic_gam_model', 'high_earner_probability', 'logistic_gam_auc']
