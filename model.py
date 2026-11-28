"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    path  = os.path.join(tempfile.gettempdir(), "Wage,csv")
    
    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)

def describe_wage(df):
    return {
        "n": len(df),
        "columns": list(df.columns),
        "age_range":(int(df["age"].min()), int(df["age"].max())),
        "wage_mean": round(df["wage"].mean(), 2),
        "n_education_levels": df["education"].nunique()
    }

# Step 2 - split_wage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size = test_size,
        random_state = random_state
    )
    return train ,test

def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y

def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })

# Step 3 - cv_tools
import numpy as np
from sklearn.model_selection import cross_val_score


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

# Step 4 - polynomial_regression
import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


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
import numpy as np
import pandas as pd


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
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

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
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

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

# Step 10 - gam_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


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
import numpy as np
import pandas as pd


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({"age": np.linspace(lo, hi, n)})


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

# Step 12 - logistic_gam (not yet solved)
# TODO: implement

# Step 13 - fit_age_models (not yet solved)
# TODO: implement

# Step 14 - test_comparison (not yet solved)
# TODO: implement

