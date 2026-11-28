"""Functions for data in the splines and GAM walkthrough."""

import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1 - load_wage

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

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

__all__ = ['load_wage', 'describe_wage', 'split_wage', 'age_xy', 'age_grid']
