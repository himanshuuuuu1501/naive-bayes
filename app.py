import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("Naive Bayes Classifier")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Target selection
    target_col = st.selectbox("Select Target Column", df.columns)

    # Feature selection
    feature_cols = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_col]
    )

    # Train-test split
    split = st.slider("Train-Test Split (%)", 50, 90, 80)

    if st.button("Evaluate Model"):

        if len(feature_cols) == 0:
            st.error("Please select at least one feature.")
        else:

            X = df[feature_cols]
            y = df[target_col]

            # Handle categorical features
            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=(100-split)/100,
                random_state=42
            )

            model = GaussianNB()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.success("Model Evaluated Successfully")

            st.write("### Train Accuracy:", accuracy_score(y_train, train_pred))
            st.write("### Test Accuracy:", accuracy_score(y_test, test_pred))

            st.subheader("Train Confusion Matrix")
            st.write(confusion_matrix(y_train, train_pred))

            st.subheader("Test Confusion Matrix")
            st.write(confusion_matrix(y_test, test_pred))