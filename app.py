"""Interactive Streamlit app for the splines and GAMs project."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import KFold, StratifiedKFold

from beyond_linearity import (
    age_grid,
    age_xy,
    cv_mse,
    education_effect,
    fit_age_models,
    gam_model,
    gam_test_rmse,
    gam_xy,
    high_earner,
    high_earner_probability,
    logistic_gam_auc,
    logistic_gam_model,
    load_wage,
    local_model,
    partial_effect,
    poly_model,
    smooth_model,
    spline_model,
    split_wage,
    step_model,
    test_rmse,
)


st.set_page_config(
    page_title="Beyond Linearity",
    page_icon="〰️",
    layout="wide",
)


@st.cache_data(show_spinner="Loading the ISLP Wage data…")
def get_data():
    data = load_wage()
    train, test = split_wage(data)
    return data, train, test


def build_age_model(name, setting, sample_size):
    if name == "Linear regression":
        return poly_model(1)
    if name == "Polynomial regression":
        return poly_model(int(setting))
    if name == "Step function":
        return step_model(int(setting))
    if name == "Regression spline":
        return spline_model(int(setting))
    if name == "Smoothing spline":
        return smooth_model(float(setting))
    return local_model(float(setting), sample_size)


@st.cache_resource(show_spinner="Fitting the selected model…")
def fit_age_model(name, setting, train):
    X, y = age_xy(train)
    model = build_age_model(name, setting, len(X))
    return model.fit(X, y)


@st.cache_data(show_spinner=False)
def score_age_model(name, setting, train):
    X, y = age_xy(train)
    model = build_age_model(name, setting, len(X))
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    return cv_mse(model, X, y, cv)


@st.cache_resource(show_spinner="Fitting the additive model…")
def fit_additive_model(train):
    X, y = gam_xy(train)
    return gam_model().fit(X, y)


@st.cache_resource(show_spinner="Comparing models on the training data…")
def fit_comparison_models(train):
    X, y = age_xy(train)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    return fit_age_models(X, y, cv)


@st.cache_resource(show_spinner="Fitting the high-earner model…")
def fit_high_earner_model(train):
    X, _ = gam_xy(train)
    target = high_earner(train)
    return logistic_gam_model().fit(X, target)


def show_curve_chart(train, model, title):
    X, _ = age_xy(train)
    grid = age_grid()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(
        X["age"], train["wage"],
        s=10, alpha=0.17, color="#63756b", label="Training rows",
    )
    ax.plot(grid["age"], model.predict(grid), color="#327451", linewidth=2.6, label=title)
    ax.set(xlabel="Age", ylabel="Wage (thousands of dollars)", title="How the selected model fits age")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.title("Beyond Linearity")
st.write(
    "Explore how polynomials, splines, and generalized additive models "
    "describe changing patterns in the ISLP Wage data."
)
st.caption(
    "This is a learning tool. The fitted relationships are associations, "
    "not proof that age, year, or education causes a wage change."
)

try:
    data, train, test = get_data()
except Exception:
    st.error("The Wage data could not be loaded. Check the app logs and try again.")
    st.stop()

st.sidebar.header("Project data")
st.sidebar.write(f"**{len(data):,}** Wage records")
st.sidebar.write(f"Training set: {len(train):,} rows")
st.sidebar.write(f"Held-out set: {len(test):,} rows")
st.sidebar.caption("The Wage file is downloaded from the ISLP data repository on first use.")

age_tab, gam_tab, compare_tab, classification_tab, about_tab = st.tabs(
    ["Explore age curves", "Understand a GAM", "Compare models", "High-earner probability", "About"]
)

with age_tab:
    st.subheader("Choose a curve and see how it fits")
    st.write(
        "Change the model and its setting. The curve is fit using the training rows; "
        "the held-out set stays separate for the comparison tab."
    )

    model_name = st.selectbox(
        "Model",
        [
            "Linear regression",
            "Polynomial regression",
            "Step function",
            "Regression spline",
            "Smoothing spline",
            "Local smoother",
        ],
    )
    if model_name == "Polynomial regression":
        setting = st.slider("Polynomial degree", 2, 6, 4)
        explanation = "A single curve whose overall shape is controlled by its degree."
    elif model_name == "Step function":
        setting = st.select_slider("Number of age groups", options=[2, 4, 8, 16], value=8)
        explanation = "Divides age into groups and fits a separate average level for each group."
    elif model_name == "Regression spline":
        setting = st.slider("Number of knots", 3, 12, 5)
        explanation = "Joins smooth polynomial sections so the curve can bend locally."
    elif model_name == "Smoothing spline":
        setting = st.select_slider(
            "Smoothing strength",
            options=[0.001, 0.1, 1.0, 10.0, 100.0, 1000.0],
            value=10.0,
            format_func=lambda value: f"{value:g}",
        )
        explanation = "A larger penalty makes the fitted curve smoother."
    elif model_name == "Local smoother":
        setting = st.slider("Neighborhood span", 0.02, 0.70, 0.20, 0.01)
        explanation = "Predicts from nearby ages; a larger span averages over more rows."
    else:
        setting = 1
        explanation = "Fits one straight trend across the full age range."

    st.caption(explanation)
    fitted = fit_age_model(model_name, setting, train)
    show_curve_chart(train, fitted, model_name)

    col_prediction, col_score = st.columns(2)
    with col_prediction:
        selected_age = st.slider("Try an age", 18, 80, 40, key="age_prediction")
        estimate = float(fitted.predict(pd.DataFrame({"age": [selected_age]}))[0])
        st.metric("Estimated wage", f"${estimate:,.1f}k")
    with col_score:
        st.write("**Training-only check**")
        if st.button("Calculate 5-fold cross-validation error", key="score_selected_model"):
            mean_mse, standard_error = score_age_model(model_name, setting, train)
            st.metric("Cross-validation MSE", f"{mean_mse:,.1f}")
            st.caption(f"Fold standard error: {standard_error:,.1f}. Lower MSE is better.")
        else:
            st.caption("Run cross-validation to see how this setting performs on training folds.")

with gam_tab:
    st.subheader("Age, year, and education each add a contribution")
    st.write(
        "A generalized additive model (GAM) gives age and year their own smooth curves "
        "and represents education with category adjustments."
    )
    gam = fit_additive_model(train)
    X_gam, _ = gam_xy(train)
    education_levels = sorted(X_gam["education"].unique())
    year_levels = sorted(int(year) for year in X_gam["year"].unique())

    col_age, col_year, col_education = st.columns(3)
    with col_age:
        gam_age = st.slider("Age", 18, 80, 40, key="gam_age")
    with col_year:
        gam_year = st.select_slider("Year", options=year_levels, value=year_levels[len(year_levels) // 2])
    with col_education:
        gam_education = st.selectbox("Education", education_levels)

    prediction_row = pd.DataFrame(
        {"age": [gam_age], "year": [gam_year], "education": [gam_education]}
    )
    gam_estimate = float(gam.predict(prediction_row)[0])
    st.metric("GAM estimated wage", f"${gam_estimate:,.1f}k")

    age_values = age_grid()["age"]
    age_effect = partial_effect(gam, X_gam, "age", age_values)
    year_effect = partial_effect(gam, X_gam, "year", year_levels)
    edu_effect = education_effect(gam, X_gam)
    effect_col1, effect_col2 = st.columns(2)
    with effect_col1:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(age_values, age_effect, color="#327451", linewidth=2.5)
        ax.axhline(0, color="#9aa69e", linewidth=1)
        ax.set(title="Age contribution", xlabel="Age", ylabel="Centered wage effect")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with effect_col2:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(list(edu_effect), list(edu_effect.values()), color="#d98a61")
        ax.set(title="Education adjustments", xlabel="Centered wage effect")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption("Partial effects are centered for display; other inputs are held at typical values.")

with compare_tab:
    st.subheader("Compare models on rows they did not train on")
    st.write(
        "Each age-only model chooses its setting with cross-validation on the training set. "
        "The test set is used here for the final comparison."
    )
    if st.button("Run the full comparison", type="primary"):
        with st.spinner("Selecting and fitting models on the training data…"):
            selected_models = fit_comparison_models(train)
            age_X_test, age_y_test = age_xy(test)
            errors = test_rmse(selected_models, age_X_test, age_y_test)
            additive_error = gam_test_rmse(train, test)

        rows = [
            {
                "Model": name,
                "Test RMSE (thousands of dollars)": errors[name],
                "Selected setting": setting,
            }
            for name, (_, setting) in selected_models.items()
        ]
        rows.append(
            {"Model": "GAM", "Test RMSE (thousands of dollars)": additive_error, "Selected setting": "age + year + education"}
        )
        results = pd.DataFrame(rows).sort_values(
            "Test RMSE (thousands of dollars)", ignore_index=True
        )
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.caption("Lower RMSE means smaller prediction errors on this one held-out split.")
    else:
        st.info("Run the comparison when you’re ready. Model selection may take a little while.")

with classification_tab:
    st.subheader("Estimate the chance of a wage above $250k")
    st.write(
        "The logistic GAM turns the same age, year, and education inputs into a probability "
        "for the project’s high-earner threshold."
    )
    high_model = fit_high_earner_model(train)
    X_high, _ = gam_xy(train)
    high_ages = np.arange(25, 66, 5)
    probabilities = high_earner_probability(high_model, X_high, high_ages)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(high_ages, probabilities, marker="o", color="#6260a8", linewidth=2.5)
    ax.set(
        xlabel="Age",
        ylabel="Estimated probability",
        ylim=(0, 1),
        title="High-earner probability across ages",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if st.button("Calculate 5-fold AUC", key="calculate_auc"):
        with st.spinner("Scoring the classifier across five folds…"):
            target = high_earner(train)
            auc = logistic_gam_auc(X_high, target, StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
        st.metric("Cross-validation AUC", f"{auc:.3f}")
    st.caption("The chart holds year and education at typical training-set values.")

with about_tab:
    st.subheader("What this project teaches")
    st.markdown(
        """
        - **Polynomials and steps:** simple ways to let a model depart from a straight line.
        - **Splines:** smooth curves built from connected polynomial pieces.
        - **Smoothing and local fits:** control how much the curve follows the individual rows.
        - **GAMs:** add readable effects from multiple inputs.
        - **Cross-validation and a held-out set:** choose settings first, then check prediction error.

        Wage records are observational. The patterns shown here do not establish that a feature causes a wage difference, and this app should not be used for employment or pay decisions.
        """
    )
    st.link_button(
        "View the project source on GitHub",
        "https://github.com/aditya00801/beyond-linearity-splines-and-gams-with-scikit-learn",
    )
