import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv('data/train/sales_forecasting_train.csv')
test_data = pd.read_csv('data/test/sales_forecasting_test.csv')
sales_data = pd.read_csv('data/raw/synthetic_sales_data.csv')

numerical_features = sales_data.select_dtypes(include=['int64', 'float64']).columns

# Load images
rel_plot_img = 'images/rel_plot.png'
prediction_vs_ground_truth_img = 'images/prediction_vs_ground_truth.png'
eval_results_img = 'images/model_eval_results_gcp.png'
architecture = 'images/final_project_architecture.png'

# Set Streamlit app title
st.title("🌟 AutoML Hierarchical Forecasting with Vertex AI")

# Sidebar for navigation
st.sidebar.title("🚀 The Journey")
page = st.sidebar.radio("Go to", ["Introduction", "Exploratory Data Analysis (EDA)", "Model Training in Vertex AI", "Model Results", "Batch Predictions"])

# Problem Statement
if page == "Introduction":
    st.header("📖 Introduction")
    st.write("""
       This app demonstrates the AutoML Hierarchical Forecasting project using Google Cloud's Vertex AI. 
       The project involves building a sales forecasting model using regression to predict future sales trends.
       Navigate through the sections to explore the EDA results, model training and predictions, and the final model's performance.
       """)

    st.header("❓ Problem Statement")
    st.write("""
    In today's dynamic retail environment, accurately forecasting sales is crucial for optimizing inventory levels, 
    minimizing stockouts, and maximizing revenue. This project addresses the challenge of predicting future sales 
    across multiple products and store locations, accounting for various factors such as seasonality, product type, 
    and store location.

    The goal of this project is to build a robust machine learning model using Google Cloud's Vertex AI AutoML, 
    leveraging historical sales data to forecast future sales. The model will assist retailers in making data-driven 
    decisions to improve operational efficiency and profitability.
    """)

    st.header("Architecture")
    st.image(architecture, use_column_width=True)

# EDA
if page == "Exploratory Data Analysis (EDA)":
    st.header("🔍 Exploratory Data Analysis (EDA)")

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

    # Summary statistics
    st.write("### 📊 Summary Statistics")
    st.dataframe(sales_data.describe())

    # Check for missing values
    st.write("### 🚨 Missing Values")
    st.write(sales_data.isnull().sum())

    # Sales Distribution
    st.write("### 📈 Distribution of Sales Data")
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(sales_data['sales'], kde=True)
    st.pyplot(plt)

    # Boxplots for numerical features
    st.write("### 📦 Boxplots for Numerical Features")
    numerical_features = sales_data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=train_data[numerical_features])
    plt.xticks(rotation=45)
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
    st.write("### 📅 Sales Data Over Time by Product and Store")
    st.image(rel_plot_img, caption='Sales Data Over Time', use_column_width=True)

    # Markdown description of the relplot
    st.markdown("""
        The seaborn `relplot` shown in the image is a multi-faceted line plot that visualizes the `sales` over time for different `product_at_store` combinations, grouped by `product_category` (Snow and Water). The plot provides several insights:

        ### Key Observations:

        1. **Seasonal Patterns**:
           - Both `Snow` and `Water` product categories exhibit strong seasonal patterns, with sales peaking and dipping in a cyclical manner. 
           - The `Snow` category shows peaks during the winter months (as expected), indicating higher sales of snow-related products during this period.
           - The `Water` category shows a similar cyclical pattern, likely with peaks in warmer months when water-related products are in higher demand.

        2. **Product-Specific Trends**:
           - Different `product_at_store` combinations (as shown by the various colored lines) have varying sales patterns, but all follow the overall seasonal trend. This suggests that while all products within a category are influenced by seasonality, there are differences in their specific performance.
           - Some products consistently have higher sales (those with higher peaks), while others have lower sales across the same periods.

        3. **Store Location Impact**:
           - The line styles differentiate between different store locations (`Flagship`, `Suburbs`, and `Outskirts`). The patterns suggest that the location of the store also affects the sales trends:
             - Products in `Flagship` stores seem to have consistently higher sales compared to those in `Suburbs` and `Outskirts`.
             - This indicates that store location plays a significant role in sales performance, with central or flagship locations likely attracting more customers.

        4. **Yearly Comparison**:
           - The plot spans multiple years, allowing for a comparison of the same periods across different years. The consistency of the patterns across years confirms the strong seasonal influence on sales.

        5. **Category Differences**:
           - There is a visual difference between the `Snow` and `Water` categories in terms of the timing and magnitude of peaks. This further emphasizes the seasonal dependence of each product category and the importance of timing in sales strategies.

        ### Interpretation:

        - **Seasonality**: The strong seasonal patterns suggest that forecasting models must account for seasonality to accurately predict sales for these products. The cyclic nature is evident for both Snow and Water products, aligned with their respective seasonal demand.

        - **Store Strategy**: The impact of store location on sales performance should be considered in planning and inventory management. Flagship stores outperform others, which may inform decisions about stock distribution and promotional efforts.

        - **Product-Specific Insights**: Some products perform consistently better than others within the same category. This information could guide marketing and sales strategies to focus on high-performing products during peak seasons.

        - **Sales Optimization**: Understanding the specific timing and intensity of these sales peaks can help in optimizing inventory, staffing, and marketing efforts to align with expected demand surges.

        ### Actionable Insights:
        - **Seasonal Planning**: Plan inventory and marketing strategies around these seasonal peaks to maximize sales.
        - **Store-Specific Strategies**: Consider enhancing the product mix or promotional efforts in lower-performing store locations to boost sales.
        - **Product Focus**: Focus on high-performing products during their peak seasons for better returns.

        This `relplot` effectively visualizes the complex interplay between time, product type, store location, and sales, offering valuable insights into how to manage and optimize sales strategies.
        """)




# Model Cretion in Vertex AI
if page == "Model Training in Vertex AI":
    st.header("🛠️ Model Training in Vertex AI")


    st.write("""
        The Vertex AI AutoML Forecasting job was configured to build a sales prediction model by leveraging historical sales data.

        ### Key Configuration Details:

        - **Optimization Objective**: The model was trained to minimize Root Mean Squared Error (RMSE), focusing on reducing large prediction errors.

        - **Time Series Setup**: The model used a 30-day context window to predict the next 30 days, capturing short-term trends and seasonal patterns.

        - **Feature Importance**: Key features such as `store`, `product_type`, and `date` were utilized to enhance the model's accuracy, with `store` being the most influential predictor of sales.

        - **Hierarchical Grouping**: The model also leveraged hierarchical grouping by product to make more robust predictions across different product categories.

        This setup allowed the model to effectively capture the temporal dynamics and key factors influencing sales, resulting in accurate and actionable forecasts.
        """)

# Model Results - Model Evaluation Section
if page == "Model Results":
    st.header("📊 Model Results and Evaluation")

    st.write("### 🔍 Prediction vs Ground Truth")
    st.image(eval_results_img, caption='Prediction vs Ground Truth', use_column_width=True)

    st.write("### 📈 Model Evaluation")
    st.write("""
    The Vertex AI model used regression to predict sales. The model captured the trend and seasonality 
    effectively but showed some discrepancies in magnitude for certain products and store locations.
    """)

    # Key Performance Metrics
    st.write("### Key Performance Metrics")
    st.write("""
    - **Mean Absolute Error (MAE):** 1.327  
      On average, the model's predictions are off by 1.327 units of sales.

    - **Mean Absolute Percentage Error (MAPE):** 5.732  
      The model's predictions are off by approximately 5.732% relative to the actual sales values.

    - **Root Mean Squared Error (RMSE):** 1.93  
      The RMSE indicates that the model's predictions have an average squared difference from the actual values of 1.93 units.

    - **Root Mean Squared Logarithmic Error (RMSLE):** 0.074  
      RMSLE penalizes under-predictions more than over-predictions, making it useful when the target variable can have large variations or outliers.

    - **R-Squared (R²):** 0.992  
      The model explains 99.2% of the variance in the sales data, indicating very high explanatory power.
    """)

    # Feature Importance
    st.write("### Feature Importance")
    st.write("""
    The model identified the following features as the most important for predicting sales:

    - **Store:** The most significant feature, contributing over 50% to the model's decision-making process. This indicates that store location or type is crucial for sales predictions.

    - **Product Type:** Contributes around 20%, highlighting the importance of the type of product (e.g., seasonal products) in sales forecasting.

    - **Date:** Contributes about 10%, reflecting the importance of time-related patterns like seasonality.

    - **Product and Product Category:** These features are less important but still contribute to the model's predictions, indicating some influence of specific product characteristics on sales.
    """)

    # Interpretation and Recommendations
    st.write("### Interpretation and Recommendations")
    st.write("""
    - **High Model Accuracy:** The model's low MAE, MAPE, and RMSE, coupled with a high R², indicate strong predictive performance and accuracy.

    - **Focus on Store-Level Strategies:** Given the high importance of the `store` feature, retailers should prioritize store-level strategies, such as optimizing inventory and tailoring marketing efforts.

    - **Product and Seasonal Strategy:** The significance of `product_type` and `date` suggests that different products have varying sales patterns over time, which should inform inventory and marketing plans.

    - **Further Model Improvement:** Although the model performs well, further fine-tuning could enhance prediction accuracy, particularly for specific products or underrepresented stores.
    """)


# Batch Predictions
if page == "Batch Predictions":
    st.header("📦 Batch Predictions")
    st.image(prediction_vs_ground_truth_img, caption='Batch Predictions Process in Vertex AI', use_column_width=True)
    st.write("""
        The comparison between predicted and actual sales shows that the model generally captures the overall seasonal and temporal trends effectively.

        ### Key Observations:

        - **Pattern Consistency**: Predicted and actual sales follow similar patterns, indicating the model’s strength in capturing trends.

        - **Magnitude Differences**: There are some discrepancies in the magnitude of predictions, where certain products, like `Skis` and `SwimSuit`, are either slightly overestimated or underestimated.

        - **Product-Specific Accuracy**: Prediction accuracy varies across different products, with some showing closer alignment to actual sales than others.

        - **Store Location Impact**: Sales trends across different store locations are mostly consistent, though some variation suggests potential overfitting or underfitting in specific locations.

        ### Interpretation:

        The model is strong in identifying overall trends and maintaining temporal consistency, but could benefit from further fine-tuning to improve the accuracy of sales magnitude predictions, particularly for specific products and store locations.

        ### Actionable Steps:

        - **Model Fine-Tuning**: Adjust hyperparameters and consider more sophisticated models to refine prediction accuracy.
        - **Feature Engineering**: Incorporate additional features or focus on product and location-specific modeling to reduce discrepancies.
        - **Post-Modeling Analysis**: Perform residual analysis to better understand where the model is underperforming and implement necessary adjustments.
        """)

# Footer
st.write("---")
st.markdown("""
    **Developed by Ramon Kidd**  
    Github Project Repo: [Project Repo](https://github.com/imarri01/vertexai-automl-forecasting)
    Powered by [Streamlit](https://streamlit.io) and [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai).
""")
