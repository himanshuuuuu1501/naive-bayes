import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)

st.title("ML Model Evaluator")

# ---------------- Upload Dataset ----------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- Problem Type ----------------
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # ---------------- Target ----------------
    target_col = st.selectbox("Select Target Column", df.columns)

    # ---------------- Feature Selection ----------------
    # Remove target + ID column automatically
    available_features = [
        c for c in df.columns
        if c != target_col and c.lower() != "id"
    ]

    feature_cols = st.multiselect(
        "Select Feature Columns",
        available_features
    )

    # ---------------- Train / Test Split ----------------
    train_percent = st.slider("Train Percentage (%)", 50, 90, 80)
    test_percent = 100 - train_percent

    st.write(f"Train: {train_percent}% | Test: {test_percent}%")

    # ---------------- Evaluate Button ----------------
    if st.button("Evaluate Model"):

        if len(feature_cols) == 0:
            st.error("Please select at least one feature")
            st.stop()

        X = df[feature_cols]
        y = df[target_col]

        # Remove missing values
        data = pd.concat([X, y], axis=1).dropna()

        X = data[feature_cols]
        y = data[target_col]

        # Convert categorical features
        X = pd.get_dummies(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=42
        )

        # ==================================================
        # CLASSIFICATION
        # ==================================================
        if problem_type == "Classification":

            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

            model = GaussianNB()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.success("Classification Model Evaluated")

            st.write("Train Accuracy:",
                     round(accuracy_score(y_train, train_pred), 4))
            st.write("Test Accuracy:",
                     round(accuracy_score(y_test, test_pred), 4))

            # SMALL NORMAL CONFUSION MATRIX
            st.subheader("Train Confusion Matrix")
            st.write(pd.DataFrame(
                confusion_matrix(y_train, train_pred)
            ))

            st.subheader("Test Confusion Matrix")
            st.write(pd.DataFrame(
                confusion_matrix(y_test, test_pred)
            ))

        # ==================================================
        # REGRESSION
        # ==================================================
        else:

            model = LinearRegression()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.success("Regression Model Evaluated")

            st.write("Train R² Score:",
                     round(r2_score(y_train, train_pred), 4))
            st.write("Test R² Score:",
                     round(r2_score(y_test, test_pred), 4))

            st.write("Train MSE:",
                     round(mean_squared_error(y_train, train_pred), 4))
            st.write("Test MSE:",
                     round(mean_squared_error(y_test, test_pred), 4))
