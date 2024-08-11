import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv('data/train/sales_forecasting_train.csv')
test_data = pd.read_csv('data/test/sales_forecasting_test.csv')

# Load images
rel_plot_img = 'images/rel_plot.png'
prediction_vs_ground_truth_img = 'images/prediction_vs_ground_truth.png'
batch_prediction_img = 'images/batch_prediction_screenshot.png'

# Set Streamlit app title
st.title("AutoML Hierarchical Forecasting with Vertex AI")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Problem Statement", "Introduction", "EDA", "Model Results", "Batch Predictions"])

# Problem Statement
if page == "Problem Statement":
    st.header("Problem Statement")
    st.write("""
    In today's dynamic retail environment, accurately forecasting sales is crucial for optimizing inventory levels, 
    minimizing stockouts, and maximizing revenue. This project addresses the challenge of predicting future sales 
    across multiple products and store locations, accounting for various factors such as seasonality, product type, 
    and store location.

    The goal of this project is to build a robust machine learning model using Google Cloud's Vertex AI AutoML, 
    leveraging historical sales data to forecast future sales. The model will assist retailers in making data-driven 
    decisions to improve operational efficiency and profitability.
    """)

# Introduction
if page == "Introduction":
    st.header("Introduction")
    st.write("""
    This app demonstrates the AutoML Hierarchical Forecasting project using Google Cloud's Vertex AI. 
    The project involves building a sales forecasting model using regression to predict future sales trends.
    Navigate through the sections to explore the EDA results, model training and predictions, and the final model's performance.
    """)

# EDA
if page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Show first few rows of the data
    st.write("### Sample of the Training Data")
    st.dataframe(train_data.head())

    # Display the basic information about the dataset
    st.write("### Basic Information of the Dataset")
    buffer = []
    train_data.info(buf=buffer)
    s = '\n'.join(buffer)
    st.text(s)

    # Summary statistics
    st.write("### Summary Statistics")
    st.dataframe(train_data.describe())

    # Check for missing values
    st.write("### Missing Values")
    st.write(train_data.isnull().sum())

    # Sales Distribution
    st.write("### Distribution of Sales Data")
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['sales'], kde=True)
    st.pyplot(plt)

    # Boxplots for numerical features
    st.write("### Boxplots for Numerical Features")
    numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=train_data[numerical_features])
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    correlation_matrix = train_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    # Sales Data Over Time by Product and Store
    st.write("### Sales Data Over Time by Product and Store")
    st.image(rel_plot_img, caption='Sales Data Over Time', use_column_width=True)

# Model Results
if page == "Model Results":
    st.header("Model Training and Results")

    st.write("### Prediction vs Ground Truth")
    st.image(prediction_vs_ground_truth_img, caption='Prediction vs Ground Truth', use_column_width=True)

    st.write("### Model Evaluation")
    st.write("""
    The Vertex AI model used regression to predict sales. The model captured the trend and seasonality 
    effectively but showed some discrepancies in magnitude for certain products and store locations.
    """)

    st.write("### Key Performance Metrics")
    st.write("""
    - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors between predicted and actual sales. 
    - **Mean Absolute Error (MAE)**: Indicates the average absolute difference between predicted and actual values.
    - **R-Squared (R²)**: Reflects how well the model explains the variability of the target variable (sales).

    Based on the model development notebook, the key metrics were as follows:
    - **RMSE**: [Insert RMSE Value from Notebook]
    - **MAE**: [Insert MAE Value from Notebook]
    - **R-Squared**: [Insert R-Squared Value from Notebook]
    """)

    st.write("### Model Interpretation")
    st.write("""
    - The model performed well in capturing overall trends and seasonality but had some challenges with accurately predicting sales for specific products and locations.
    - Certain products exhibited higher variance in sales, which the model struggled to predict precisely.
    - Store location played a significant role in prediction accuracy, with flagship stores generally showing more accurate predictions compared to suburban or outskirts locations.

    Further fine-tuning and model adjustments could help improve accuracy, especially for outlier products and underperforming locations.
    """)

# Batch Predictions
if page == "Batch Predictions":
    st.header("Batch Predictions")
    st.image(batch_prediction_img, caption='Batch Predictions Process in Vertex AI', use_column_width=True)
    st.write("""
    This section demonstrates the batch prediction process using the trained model in Vertex AI. 
    The model scales efficiently, handling batch predictions across various product categories and store locations.
    """)

# Footer
st.write("---")
st.write("Developed by Ramon Kidd. Powered by Streamlit and Google Cloud Vertex AI.")
