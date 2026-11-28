# Beyond Linearity: Splines and GAMs with scikit-learn

This project follows the beyond-linearity chapter of *An Introduction to Statistical Learning* using the book's Wage data. It compares polynomial and step-function fits, regression and smoothing splines, a local smoother, and generalized additive models (GAMs).

## Setup

Use Python 3.11 or later. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the walkthrough

```bash
python scaffold.py
```

The first run downloads `Wage.csv` from the ISLP data repository and caches it in the system temporary directory. The script prints the cross-validation comparisons, extrapolation examples, partial effects, and test-set errors to the terminal.

## Use the interactive app

**Try it online:** [Open the live Streamlit app](https://beyond-linearity-splines-and-gams-with-scikit-learn-w8eqe5c4zv.streamlit.app/).

```bash
streamlit run app.py
```

Choose a model, adjust its flexibility, view the fitted curve, inspect GAM effects, and run the held-out comparison. The app uses the same functions as the command-line walkthrough.

## Deploy your own copy

The project already has a [live app](https://beyond-linearity-splines-and-gams-with-scikit-learn-w8eqe5c4zv.streamlit.app/). To deploy your own copy, use [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with GitHub and connect the public repository.
2. Create an app from `aditya00801/beyond-linearity-splines-and-gams-with-scikit-learn`, branch `main`, entrypoint `app.py`.
3. Deploy it and share the URL Streamlit assigns.

Streamlit Community Cloud needs permission to read the GitHub repository. Its deployment steps are described in the [official guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Run the tests

```bash
python -m unittest discover -s tests
```

The tests use small synthetic datasets and do not download the Wage data.

## Project layout

```text
beyond_linearity/       Reusable implementation, grouped by topic
  data.py               Wage data, train/test split, and age grids
  validation.py         Cross-validation and one-standard-error rule
  models.py             Polynomial, step, spline, and local models
  additive.py           GAMs and partial-effect helpers
  evaluation.py         Model comparison and test-set errors
tests/                  Offline unit tests
app.py                  Interactive Streamlit app
docs/                   Project overview page
model.py                Compatibility imports for earlier examples
scaffold.py             Command-line walkthrough
requirements.txt        Runtime dependencies
```

The original `model.py` imports remain available for existing examples. New code can import functions from `beyond_linearity` or its topic modules.
