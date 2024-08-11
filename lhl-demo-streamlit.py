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
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model Results", "Batch Predictions"])

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

    st.write("### Distribution of Sales Data")
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['sales'], kde=True)
    st.pyplot(plt)

    st.write("### Sales Data Over Time by Product and Store")
    st.image(rel_plot_img, caption='Sales Data Over Time', use_column_width=True)

    st.write("### Correlation Heatmap")
    correlation_matrix = train_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

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
st.write("Developed by [Your Name]. Powered by Streamlit and Vertex AI.")
