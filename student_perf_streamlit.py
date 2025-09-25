import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Student Performance Prediction App")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your Student Performance CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())
    st.write(df.tail())

    # Splitting features and target
    if "Performance Index" in df.columns:
        X = df.drop("Performance Index", axis=1)
        y = df["Performance Index"]

        # Encoding categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Show results
        st.subheader("Model Performance")
        st.write("R-squared Score:", model.score(X_test, y_test))

        # Plot Actual vs Predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Performance Index")
        ax.set_ylabel("Predicted Performance Index")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    else:
        st.error("The uploaded file does not contain 'Performance Index' column.")