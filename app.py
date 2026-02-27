import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)

st.title("ML Model Evaluator with EDA")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==================================================
    # EDA SECTION
    # ==================================================
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Matrix")
    st.write(df.select_dtypes(include=["number"]).corr())

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.subheader("Histogram")
        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna())
        ax.set_title(f"Histogram of {selected_col}")
        st.pyplot(fig)

    # ==================================================
    # MODEL SECTION
    # ==================================================
    st.header("Model Training")

    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # ==================================================
    # AUTO TARGET SELECTION
    # ==================================================
    if problem_type == "Classification":

        if "Species" in df.columns:
            target_col = "Species"
            st.write("Target Column (Auto Selected): **Species**")
        else:
            st.error("Species column not found")
            st.stop()

    else:

        if "Id" in df.columns:
            target_col = "Id"
            st.write("Target Column (Auto Selected): **Id**")
        else:
            st.error("Id column not found")
            st.stop()

    # ---------------- Features ----------------
    feature_cols = st.multiselect(
        "Select Feature Columns",
        [c for c in df.columns if c != target_col]
    )

    # ---------------- Train/Test ----------------
    train_percent = st.slider("Train Percentage (%)", 50, 90, 80)
    test_percent = 100 - train_percent

    st.write(f"Train: {train_percent}% | Test: {test_percent}%")

    # ---------------- Evaluate ----------------
    if st.button("Evaluate Model"):

        if len(feature_cols) == 0:
            st.error("Select features")
            st.stop()

        X = df[feature_cols]
        y = df[target_col]

        data = pd.concat([X, y], axis=1).dropna()

        X = data[feature_cols]
        y = data[target_col]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=42,
            stratify=y if problem_type == "Classification" else None
        )

        # ================= CLASSIFICATION =================
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

            st.subheader("Train Confusion Matrix")
            st.write(pd.DataFrame(
                confusion_matrix(y_train, train_pred)
            ))

            st.subheader("Test Confusion Matrix")
            st.write(pd.DataFrame(
                confusion_matrix(y_test, test_pred)
            ))

        # ================= REGRESSION =================
        else:

            model = LinearRegression()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.success("Regression Model Evaluated")

            st.write("Train R²:",
                     round(r2_score(y_train, train_pred), 4))
            st.write("Test R²:",
                     round(r2_score(y_test, test_pred), 4))

            st.write("Train MSE:",
                     round(mean_squared_error(y_train, train_pred), 4))
            st.write("Test MSE:",
                     round(mean_squared_error(y_test, test_pred), 4))
