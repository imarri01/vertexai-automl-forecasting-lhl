import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vertexai.preview import batch_prediction

# Load datasets
train_data = pd.read_csv('data/train/sales_forecasting_train.csv')
test_data = pd.read_csv('data/test/sales_forecasting_test.csv')
sales_data = pd.read_csv('data/raw/synthetic_sales_data.csv')


numerical_features = sales_data.select_dtypes(include=['int64', 'float64']).columns

# Load images
rel_plot_img = 'images/rel_plot.png'
prediction_vs_ground_truth_img = 'images/prediction_vs_ground_truth.png'
eval_results_img = 'images/model_eval_results_gcp.png'
architecture = 'images/final_architecture.png'
storefront = 'images/storefront.jpeg'
batch_prediction = 'images/batch_prediction_screenshot.png'
vertex_code = 'images/vertex_code_screenshot.png'

# Set Streamlit app title
st.title("🌟 AutoML Hierarchical Forecasting with Vertex AI")

# Sidebar for navigation
st.sidebar.title("🚀 The Journey")
page = st.sidebar.radio("Go to", ["Introduction", "The Architecture", "The Data", "Model Training in Vertex AI", "Model Results", "Batch Predictions"])

# Problem Statement
if page == "Introduction":
    st.header("📖 ❓ Intro & Problem Statement")
    st.image(storefront, use_column_width=True)

if page == "The Architecture":
    st.header("Architecture")
    st.image(architecture, use_column_width=True)

# EDA
if page == "The Data":
    st.header("🔍 The Data")

    # Show first few rows of the data
    st.write("### 👀 Sample of the Training Data")
    st.dataframe(sales_data.head())

    # Display the basic information about the dataset
    st.write("### 🛠️ Basic Information of the Dataset")
    st.text("""
     <class 'pandas.core.frame.DataFrame'>
     RangeIndex: 21930 entries, 0 to 21929
     Data columns (total 11 columns):
      #   Column            Non-Null Count  Dtype  
     ---  ------            --------------  -----  
      0   date_index        21930 non-null  int64  
      1   product_index     21930 non-null  int64  
      2   store_index       21930 non-null  int64  
      3   date              21930 non-null  object 
      4   day_of_week       21930 non-null  int64  
      5   temperature       21930 non-null  float64
      6   product           21930 non-null  object 
      7   product_type      21930 non-null  object 
      8   product_category  21930 non-null  object 
      9   store             21930 non-null  object 
      10  sales             21930 non-null  float64
     dtypes: float64(2), int64(4), object(5)
     memory usage: 1.8+ MB
     """)


    # Sales Distribution
    st.write("### 📈 Distribution of Sales Data")
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(sales_data['sales'], kde=True)
    st.pyplot(plt)

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    correlation_matrix = sales_data[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    #plt.show()
    st.pyplot(plt)


    # Sales Data Over Time by Product and Store
    # st.write("### 📅 Sales Data Over Time by Product and Store")
    # st.image(rel_plot_img, caption='Sales Data Over Time', use_column_width=True)

    # Markdown description of the relplot
    # st.markdown("""
    #    The charts provide insights into the sales distribution and the correlations among numerical features in the dataset. The Distribution of Sales Data chart shows that the majority of sales are clustered between 10 to 30 units, with a noticeable peak around 15 units, indicating a common sales range. There are fewer instances of higher sales, with a gradual decline as sales numbers increase. The Correlation Heatmap highlights that the most significant correlation is between store_index and sales with a strong negative correlation of -0.79, suggesting that certain store locations consistently have lower sales. Additionally, day_of_week has a slight positive correlation with sales (0.18), indicating a potential weekly pattern in sales trends, while other features like temperature show minimal impact on sales. Overall, the analysis suggests that store location and the day of the week are influential factors in sales performance.
    #     """)




# Model Cretion in Vertex AI
if page == "Model Training in Vertex AI":
    st.header("🛠️ Model Training in Vertex AI")
    st.image(vertex_code, use_column_width=True)




# Model Results - Model Evaluation Section
if page == "Model Results":
    st.header("📊 Model Results and Evaluation")

    st.write("### 🔍 VertexAI model using AutoML in Google Cloud")
    st.image(eval_results_img, caption='Prediction vs Ground Truth', use_column_width=True)




# Batch Predictions
if page == "Batch Predictions":
    st.header("📦 VertexAI batch job")
    st.image(batch_prediction, use_column_width=True)

    st.header("📦 Job Results")
    st.image(prediction_vs_ground_truth_img, caption='Batch Predictions Process in Vertex AI', use_column_width=True)
    st.write("""
        The comparison between predicted and actual sales shows that the model generally captures the overall seasonal and temporal trends effectively.

      
         """)

# Footer
st.write("---")
st.markdown("""
    **Developed by Ramon Kidd**  
    Github Project Repo: [Project Repo](https://github.com/imarri01/vertexai-automl-forecasting)
    Powered by [Streamlit](https://streamlit.io) and [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai).
""")
