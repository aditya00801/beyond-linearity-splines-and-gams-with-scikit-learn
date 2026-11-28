# Beyond Linearity: Splines and GAMs with scikit-learn

This project follows the beyond-linearity chapter of *An Introduction to Statistical Learning* using the book's Wage data. It compares polynomial and step-function fits, regression and smoothing splines, a local smoother, and generalized additive models (GAMs).

## Setup

Use Python 3.9 or later. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the walkthrough

```bash
python scaffold.py
```

The first run downloads `Wage.csv` from the ISLP data repository and caches it in the system temporary directory. The script prints the cross-validation comparisons, extrapolation examples, partial effects, and test-set errors to the terminal.

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
docs/                   Project overview page
model.py                Compatibility imports for earlier examples
scaffold.py             Command-line walkthrough
requirements.txt        Runtime dependencies
```

The original `model.py` imports remain available for existing examples. New code can import functions from `beyond_linearity` or its topic modules.
