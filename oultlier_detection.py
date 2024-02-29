import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # Assuming CSV for simplicity
            return df
        except Exception as e:
            st.error(f"Error: {str(e)}")
    return None

def detect_outliers(column_data):
    # Use IQR method to detect outliers
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (column_data < lower_bound) | (column_data > upper_bound)
    return outliers

def remove_outliers(column_data, method):
    if method == "IQR":
        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
    elif method == "Z-score":
        z_scores = (column_data - column_data.mean()) / column_data.std()
        return column_data[(z_scores >= -3) & (z_scores <= 3)]
    # Add more outlier removal methods if needed

def main():
    st.title("Outlier Detection and Removal App")
    
    # Load dataset
    df = load_data()
    
    if df is not None:
        st.subheader("Preview of the dataset:")
        st.write(df.head())
        
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        selected_columns = st.multiselect("Select numeric columns for outlier detection:", numeric_columns)
        
        for column in selected_columns:
            st.subheader(f"Outlier Detection for '{column}':")
            
            # Display initial boxplot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)
            
            # Check if outliers are present
            outliers = detect_outliers(df[column])
            
            if outliers.any():
                st.warning(f"Outliers detected in column '{column}'!")
                
                # Ask user if they want to remove outliers
                remove_outliers_option = st.radio(f"Do you want to remove outliers in '{column}'?", ("Yes", "No"), key=f"{column}_remove_option")
                
                if remove_outliers_option == "Yes":
                    # Provide options for outlier removal methods
                    remove_method = st.selectbox(f"Select outlier removal method for '{column}':", ("IQR", "Z-score"), key=f"{column}_remove_method")
                    
                    # Remove outliers
                    df[column] = remove_outliers(df[column], remove_method)
                    
                    st.subheader(f"Dataset after removing outliers in '{column}':")
                    st.write(df.head())

                    # Show evidence that outliers are removed
                    st.subheader(f"Boxplot after removing outliers in '{column}':")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=df[column], ax=ax)
                    st.pyplot(fig)
                else:
                    st.info(f"Outliers in '{column}' are retained.")
            
            else:
                st.success(f"No outliers detected in column '{column}'.")

if __name__ == "__main__":
    main()
