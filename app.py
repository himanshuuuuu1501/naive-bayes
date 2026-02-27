import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("Naive Bayes Classifier")

# ---------------- Upload Dataset ----------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- Target Column ----------------
    target_col = st.selectbox("Select Target Column", df.columns)

    # ---------------- Feature Selection ----------------
    feature_cols = st.multiselect(
        "Select Feature Columns",
        [c for c in df.columns if c != target_col]
    )

    # ---------------- Train-Test Split ----------------
    split = st.slider("Train Size (%)", 50, 90, 80)

    # ---------------- Evaluate Button ----------------
    if st.button("Evaluate Model"):

        if len(feature_cols) == 0:
            st.error("Please select at least one feature")
            st.stop()

        # Prepare data
        X = df[feature_cols]
        y = df[target_col]

        # Remove missing values
        data = pd.concat([X, y], axis=1).dropna()

        X = data[feature_cols]
        y = data[target_col].astype(str)

        # Check target validity
        if y.nunique() < 2:
            st.error("Target must have at least 2 classes")
            st.stop()

        # Convert categorical features
        X = pd.get_dummies(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(100 - split) / 100,
            random_state=42
        )

        # Model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # ---------------- Results ----------------
        st.success("Model Evaluated Successfully")

        st.write("### Train Accuracy:", round(accuracy_score(y_train, train_pred), 4))
        st.write("### Test Accuracy:", round(accuracy_score(y_test, test_pred), 4))

        # ---------------- Train Confusion Matrix ----------------
        st.subheader("Train Confusion Matrix")

        cm_train = confusion_matrix(y_train, train_pred)

        fig1, ax1 = plt.subplots()
        ax1.imshow(cm_train)
        ax1.set_title("Train Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")

        st.pyplot(fig1)

        # ---------------- Test Confusion Matrix ----------------
        st.subheader("Test Confusion Matrix")

        cm_test = confusion_matrix(y_test, test_pred)

        fig2, ax2 = plt.subplots()
        ax2.imshow(cm_test)
        ax2.set_title("Test Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        st.pyplot(fig2)
