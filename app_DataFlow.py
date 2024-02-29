import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Function to load example dataset (Titanic from seaborn)
def load_example_dataset():
    return sns.load_dataset('titanic')

# Function to display isnull().sum() and sort_values() before imputation
def display_before_imputation(df):
    st.subheader("Before Imputation:")
    st.write("Missing Values:")
    st.write(df.isnull().sum().sort_values(ascending=False))

# Function to impute missing values for numeric columns
def impute_numeric_values(df, method):
    numeric_cols = df.select_dtypes(include='number').columns
    if method == 'Mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'Bfill':
        df[numeric_cols] = df[numeric_cols].bfill()
    elif method == 'Ffill':
        df[numeric_cols] = df[numeric_cols].ffill()
    return df

# Function to impute missing values for categorical columns
def impute_categorical_values(df, method):
    categorical_cols = df.select_dtypes(exclude='number').columns
    if method == 'Mode':
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    elif method == 'Bfill':
        df[categorical_cols] = df[categorical_cols].bfill()
    elif method == 'Ffill':
        df[categorical_cols] = df[categorical_cols].ffill()
    return df

# Function to handle imputation based on user input
def impute_missing_values(data, numeric_method, categorical_method):
    display_before_imputation(data)

    # Impute missing values for numeric columns
    data = impute_numeric_values(data, numeric_method)

    # Impute missing values for categorical columns
    data = impute_categorical_values(data, categorical_method)

    # Display imputed data
    st.subheader("Imputed Data:")
    st.write(data)

    # Display isnull().sum() after imputation
    st.subheader("After Imputation:")
    st.write("Remaining Missing Values:")
    st.write(data.isnull().sum().sort_values(ascending=False))
# Function to plot correlation heatmap using Plotly
def plot_correlation_heatmap(data):
    st.subheader("Correlation Heatmap:")
    numeric_columns = data.select_dtypes(include='number').columns
    correlation_matrix = data[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='viridis',  # Change this to a valid Plotly colorscale
        colorbar=dict(title='Correlation'),
        showscale=True,
    ))
    st.plotly_chart(fig)
# Main Streamlit app
def main():
    st.title("DataFlow Navigator")

    # Load example dataset or upload user's dataset
    option = st.radio("Choose an option:", ["Use Example Dataset", "Upload Your Own Dataset"])

    if option == "Use Example Dataset":
        data = load_example_dataset()
    else:
        # Upload data
        uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "tsv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error: {e}")
                return None
        else:
            st.warning("Please upload a dataset.")
            return None

    # Display basic data information
    st.subheader("Data Information:")
    st.write("Data Head:")
    st.write(data.head())
    st.write("Data Shape:", data.shape)
    st.write("Data Description:")
    st.write(data.describe())
    st.write("Column Names:")
    st.write(list(data.columns))

    # Display missing values heatmap
    st.subheader("Missing Values Heatmap:")
    missing_data_cols = data.columns[data.isnull().any()]
    if not missing_data_cols.empty:
        # Create heatmap for missing values
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[missing_data_cols].isnull(), cbar=False, ax=ax)
        plt.title("Missing Values Heatmap")
        st.pyplot(fig)
    else:
        st.write("No missing values in this dataset.")

    # Plot correlation heatmap
    plot_correlation_heatmap(data)
    # Impute missing values based on user selection
    st.subheader("Imputation:")
    numeric_method = st.selectbox("Choose Numeric Imputation Method:", ['Mean', 'Bfill', 'Ffill'])
    categorical_method = st.selectbox("Choose Categorical Imputation Method:", ['Mode', 'Bfill', 'Ffill'])

    impute_missing_values(data, numeric_method, categorical_method)

    # Select columns for plotting
    st.subheader("Select Columns for Plotting:")
    x_column = st.selectbox("Select X-axis column", list(data.columns), index=0)
    y_column = st.selectbox("Select Y-axis column", list(data.columns), index=1)

    # Choose plot type
    plot_type = st.selectbox("Select Plot Type:", ["plotly", "seaborn"])

    # Plot the graph
    st.subheader("Graph:")
    if plot_type == "plotly":
        plotly_plot(data, x_column, y_column)
    elif plot_type == "seaborn":
        seaborn_plot(data, x_column, y_column)
    else:
        st.warning("Invalid plot type selected.")

# Plotly scatter plot
def plotly_plot(data, x_column, y_column):
    st.subheader("Plotly Plot:")
    plot_type = st.selectbox("Select Plot Type:", ["scatter", "line", "bar", "histogram", "box"])
    
    if plot_type == "scatter":
        fig = px.scatter(data, x=x_column, y=y_column, title="Plotly Scatter Plot")
    elif plot_type == "line":
        fig = px.line(data, x=x_column, y=y_column, title="Plotly Line Plot")
    elif plot_type == "bar":
        fig = px.bar(data, x=x_column, y=y_column, title="Plotly Bar Plot")
    elif plot_type == "histogram":
        fig = px.histogram(data, x=x_column, title="Plotly Histogram Plot")
    elif plot_type == "box":
        fig = px.box(data, x=x_column, y=y_column, title="Plotly Box Plot")
    
    st.plotly_chart(fig)

# Seaborn scatter plot
def seaborn_plot(data, x_column, y_column):
    st.subheader("Seaborn Plot:")
    plot_type = st.selectbox("Select Plot Type:", ["scatter", "line", "bar", "histplot", "boxplot"])
    
    if plot_type == "scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)
    elif plot_type == "line":
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)
    elif plot_type == "bar":
        fig, ax = plt.subplots()
        sns.barplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)
    elif plot_type == "histplot":
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=x_column, bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    elif plot_type == "boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
