"""Functions for models in the splines and GAM walkthrough."""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer, StandardScaler

from .validation import cv_curve, one_se_rule

# Step 4 - polynomial_regression

def poly_model(degree):
    # Scale first so high powers stay numerically stable
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )

def poly_curve(X, y, degrees, cv):
    # CV mean MSE and standard error for each polynomial degree
    means, ses = cv_curve(poly_model, X, y, degrees, cv)
    return means, ses

def anova_degrees(X, y, max_degree):
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    def rss(d):
        # Residual sum of squares of a degree-d fit on all the data
        model = poly_model(d).fit(X, y)
        resid = y - model.predict(X)
        return float(np.sum(resid ** 2))

    results = []
    prev_rss = rss(1)
    for d in range(2, max_degree + 1):
        cur_rss = rss(d)
        df = n - d - 1
        # F-test of degree d against degree d - 1 (1 extra parameter)
        F = (prev_rss - cur_rss) / (cur_rss / df)
        p = f_dist.sf(F, 1, df)
        results.append((d, round(float(F), 2), round(float(p), 4)))
        prev_rss = cur_rss
    return results

def choose_degree(X, y, degrees, cv, alpha=0.05):
    # Degree with the lowest CV error
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    # Largest degree reached before the first non-significant test
    max_degree = max(max(degrees), 2)
    degree_anova = 1
    for d, F, p in anova_degrees(X, y, max_degree):
        if p < alpha:
            degree_anova = d
        else:
            break
    return degree_min, degree_anova

def curve_on_grid(model, grid):
    # Flatten predictions to 1-D and round to 2 decimals
    return np.round(np.asarray(model.predict(grid)).ravel(), 2)

# Step 5 - step_functions

def step_model(n_bins):
     # Divide data into uniform bins and apply linear regression
    return make_pipeline(      # combines both operations
        KBinsDiscretizer(
            n_bins = n_bins,                           #specifies the number of bins
            encode = 'onehot-dense',                   # converts bins into one-hot encoded features.
            strategy = 'uniform'                       # reates bins of equal width.
        ),                                             #divides continuous data into bins.
        LinearRegression() # predicts the output using the encoded bins.
    )

def step_curve(X, y, bins, cv):
    # Evaluate different bin counts using cross-validation
    return cv_curve(step_model, X, y, bins, cv)
    # it return means, ses


def choose_bins(X, y, bins, cv):
      # Find minimum-MSE and one-SE bin counts

    #calculate the cross-validation results
    means, ses = step_curve(X, y,bins, cv)

    # Find the bin count with minimum MSE
    bins_min = bins[np.argmin(means)]

    # apply one_se_rule()
    bins_1se = one_se_rule(
        bins, means, ses,
        prefer = 'smaller'
    )

    return bins_min, bins_1se



def bin_edges(model):
    # Extract fitted bin boundaries and round to 1 decimal
    edges = model.named_steps['kbinsdiscretizer'].bin_edges_[0]

    return np.round(edges,1).tolist()

def step_levels(model, grid):
     # Get sorted, distinct predictions rounded to 1 decimal
    grid = np.asarray(grid).reshape(-1,1)

    predictions = model.predict(grid)

    return sorted(
        np.round(np.unique(predictions), 1).tolist()
    )

# Step 6 - spline_regression


def spline_model(n_knots, degree=3, extrapolation='constant'):

    # Create a pipeline: SplineTransformer + LinearRegression
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,             # Number of knots
            degree=degree,               # Polynomial degree
            knots='quantile',            # Place knots using quantiles
            extrapolation=extrapolation, # Handle values outside training range
            include_bias=False           # Avoid redundant bias column
        ),
        LinearRegression()               # Fit regression on spline features
    )


def spline_basis_size(model, X):

    # Access the fitted spline transformer from the pipeline
    transformer = model.named_steps['splinetransformer']

    # Transform a few rows and return the number of basis columns
    return transformer.transform(X[:5]).shape[1]


def spline_curve(X, y, knot_counts, cv):

    # Evaluate different knot counts using cross-validation
    return cv_curve(
        spline_model,
        X,
        y,
        knot_counts,
        cv
    )

def choose_knots(X, y, knot_counts, cv):

    # Get mean CV errors and standard errors
    means, ses = spline_curve(X, y, knot_counts, cv)

    # Select the minimum-error and one-SE knot counts
    k_min = knot_counts[np.argmin(means)]
    k_1se = one_se_rule(knot_counts, means, ses, prefer='smaller')

    return k_min, k_1se

# Step 7 - extrapolation


def beyond_data(models, ages):

    # Convert ages into a DataFrame
    df = pd.DataFrame({'age': ages})

    results = {}

    # Predict using each fitted model
    for name, model in models.items():
        predictions = model.predict(df)

        # Round predictions to 1 decimal
        results[name] = np.round(predictions, 1).tolist()

    return results


def extrapolation_report(X, y, ages):

    # Create three regression models
    models = {
        'poly4': poly_model(4),
        'spline_const': spline_model(5),
        'spline_linear': spline_model(5, extrapolation='linear')
    }

    # Train all three models
    for model in models.values():
        model.fit(X, y)

    # Generate predictions
    report = beyond_data(models, ages)

    # Calculate polynomial prediction range
    report['poly4_range'] = round(
        max(report['poly4']) - min(report['poly4']), 1
    )

    return report

# Step 8 - smoothing_spline

def smooth_model(alpha, n_knots=20):
    # make curve features. Step B: fit them with a penalty
    return make_pipeline(
        SplineTransformer(n_knots = n_knots,degree = 3, knots = "quantile", include_bias = False),
        Ridge(alpha= alpha)
    )


def effective_df(model, X):
    #  Get the spline columns from the fitted transformer
    B = model[0].transform(X)

    # Center them : subtract each column's mean
    B = B - B.mean(axis = 0)

    # Comppute  B^T B
    BtB= B.T @ B

    # get alpha from the Ridge step
    alpha = model[-1].alpha

    # compute (B^T b + alpha* I )^-1  B^T B
    S = np.linalg.lstsq(BtB + alpha * np.eye(BtB.shape[0]), BtB, rcond = None)[0]

    # add one  or the intercept
    return  round(float(1 + np.trace(S)), 2)


def smooth_curve(X, y, alphas, cv):
    # # Get the error and standard error for every alpha
    return cv_curve(smooth_model, X,y, alphas, cv)

def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)

    # Alpha with the lowest error
    alpha_min = alphas[np.argmin(means)]

    # larget alpha  whoes error is still cloes to the  loest

    alpha_1se = one_se_rule(alphas, means, ses, prefer = "larger")
    return alpha_min, alpha_1se

# Step 9 - local_smoother

def local_model(span, n_train):
    # span is a fraction, so convert it to a number of neighbors
    k = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=k)

def local_curve(X, y, spans, cv):
    # cv_curve gives only span, so lambda also supplies n_train
    return cv_curve(lambda span: local_model(span, len(X)), X, y, spans, cv)

def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[np.argmin(means)]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se

def roughness(curve):
    second = np.diff(np.ravel(curve), n=2)
    return round(float(np.mean(np.abs(second))), 3)

# Step 13 - fit_age_models

def fit_age_models(X, y, cv):
    models = {}

    # Linear model
    model = poly_model(1)
    model.fit(X, y)
    models['linear'] = (model, 1)

    # Select polynomial degree using ANOVA
    _, degree = choose_degree(X, y, [1, 2, 3, 4, 5, 6], cv)
    model = poly_model(degree)
    model.fit(X, y)
    models['poly'] = (model, degree)

    # Select number of bins using one-SE rule
    _, bins = choose_bins(X, y, [2, 4, 8, 16], cv)
    model = step_model(bins)
    model.fit(X, y)
    models['step'] = (model, bins)

    # Select spline knots
    _, knots = choose_knots(X, y, [3, 4, 5, 6, 8, 12], cv)
    model = spline_model(knots)
    model.fit(X, y)
    models['spline'] = (model, knots)

    # Select smoothing alpha
    _, alpha = choose_alpha(X, y, [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0], cv)
    model = smooth_model(alpha, n_knots=20)
    model.fit(X, y)
    models['smooth'] = (model, alpha)

    # Select local regression span
    _, span = choose_span(X, y, [0.02, 0.05, 0.1, 0.2, 0.4, 0.7], cv)
    model = local_model(span, len(X))
    model.fit(X, y)
    models['local'] = (model, span)

    return models


def curves_table(models, grid):
    result = {}

    # Predict values for each model
    for name, (model, setting) in models.items():
        result[name] = curve_on_grid(model, grid)

    return pd.DataFrame(result, index=np.asarray(grid).ravel())

__all__ = ['poly_model', 'poly_curve', 'anova_degrees', 'choose_degree', 'curve_on_grid', 'step_model', 'step_curve', 'choose_bins', 'bin_edges', 'step_levels', 'spline_model', 'spline_basis_size', 'spline_curve', 'choose_knots', 'beyond_data', 'extrapolation_report', 'smooth_model', 'effective_df', 'smooth_curve', 'choose_alpha', 'local_model', 'local_curve', 'choose_span', 'roughness', 'fit_age_models', 'curves_table']
