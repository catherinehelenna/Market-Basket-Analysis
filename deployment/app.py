import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

# Navigation
st.title("Segmenting Mall Customers for Strategic Marketing Approaches")
page = st.selectbox("Select a page", ["Home", "EDA","About Model","Segmentation Result","Go Predict!"])

if page == "Home":
    st.header("Warm Welcome!")

    st.write("Welcome to the Mall Customer Segmentation Tool. This application helps you classify customers into segments based on their spending habits and preferences and choose targeted marketing strategies.")

    st.write("### Tips:")
    st.write("1. Use Select Box above to explore the data analysis, model development, segmentation result, and deployment.")
    st.write("2. Want to know how our dataset is like? Please visit the **EDA** section.")
    st.write("3. Interested in understanding the model and evaluation metrics? We got you covered in **About Model** section.")
    st.write("4. See how we interpret our clustering result in **Segmentation Result** section.")
    st.write("5. The fun part: Try **Go Predict!** yourself and see how we categorize your customer and recommended marketing approaches to your customer.")

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    # dataset loading and overview
    data = pd.read_csv("final_dataset.csv")
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    st.write("Here's a preview of the first 5 rows:")
    st.write(data.head())
    st.markdown("**About the dataset**:")
    st.write("The data is from membership cards which provide Customer ID, age, gender, annual income, and spending score. Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.")
    st.markdown("[Click here to access the dataset source](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    # show results
    st.markdown("**1. Pair Plot of Original Data**")
    # Load image into a PIL Image object
    image_pairplot = Image.open("pair-plot-gender.png")
    # Display image
    st.image(image_pairplot, caption="Based on the pairplots, we can see that spending score and annual income forms five distinct clusters indicating their importance in clustering. On the other hand, there is no visible relationship patterns based on gender and age for mall customer's spending behavior.", use_container_width=True)    

    st.markdown("**2. Age Group Categories**")
    st.write("Originally, the dataset had column age which was a continuous feature. However, during the modeling process, age_group had better clustering performance than age.")
    st.write("Therefore, age was converted into age_group with the following bins and categories.")
    categories = {'age_group':['youth','working_pop','mid_age','retired'],
                  'age_range (years old)':['< 20','21-40','41-60','61 >']}
    categories_df = pd.DataFrame(categories)
    
    st.table(categories_df)
    st.write("Where working_pop is working population and mid_age is middle aged.")

elif page == "About Model":
    st.header("About the Model")
    st.write("In this study, three clustering algorithms were chosen: K-Means, DBSCAN, and Gaussian Mixture Model. Then, the dataset were modified into four combinations and compared with the control group. Below is the details of the experimental content.")
    data_categories = {'data_number':['1','2','3','4','5 (control)'],
                  'data_content':['spending score, annual income, age, gender','spending score, annual income, age','spending score, annual income, age_group, gender','spending score, annual income, age_group','spending score, annual income']}
    st.table(data_categories)
    st.write("Based on the Silhouette Score and Davies-Bouldin Index, K-Means model trained with data content from experiment 4 performed well due to comparable cluster quality with and richer explanation than control group.")
    st.write("##### The final K-Means model configuration is **k = 5** and **random_state = 10**.")
    st.markdown("**1. Silhouette Score**")
    st.write("This metric measures similarity of a point to its cluster compared to other clusters. The score range is from -1 to +1. Score near to +1 indicates points are well-clustered.")
    # Visualization Silhouette Score
    image_silhouette = Image.open("silhouette scores.png")
    # Display image
    st.image(image_silhouette, caption="Among all experimental models, K-Means trained with dataset 4 had relatively good performance with a silhouette score of 0.44.", use_container_width=True)    

    st.markdown("**2. Davies-Bouldin Index**")
    st.write("This metric measures ratio of average intra-cluster distance to average inter-cluster distance. The smaller, the better the clustering since the separation is more apparent.")
    # Visualization DB Index
    image_dbi = Image.open("db index scores.png")
    # Display image
    st.image(image_dbi, caption="Among all experimental models,  K-Means trained with dataset 4 still had a good clustering quality with DB index of 0.88 which was less than 1.", use_container_width=True)    

elif page == "Segmentation Result":
    st.header("Customer Segmentation Result")
    st.write("After selected the best clustering model, we conducted another exploratory data analysis to characterize each cluster.")
    st.subheader("1. Cluster Profiling with Central Tendencies")
    st.markdown("**1. Mean Annual Income and Spending Scores in Age Groups**")

    # Visualization Mean
    image_barplot = Image.open("bar plot.png")
    # Display image
    st.image(image_barplot, use_container_width=True)    

    st.markdown("**2. Median Annual Income and Spending Scores in Age Groups**")
    
    # Visualization Median
    image_boxplot = Image.open("boxplot.png")

    # Display image
    st.image(image_boxplot, use_container_width=True)    

    # Customer personality data analysis
    st.write("**Conclusion**:")
    customer_profile = {
        'customer_type': [
            'High income, low spending',
            'Seniors with high income and spending',
            'Low income and low spending',
            'Young Generation with low income, high spending',
            'Medium income and spending'
        ],
        'income_range':['>$70k','>$70k','<$40k','<$40k','$40k-$70k'],
        'spending_score_range': ['<40','>60','<40','>60','40-60'],
        'personality': [
            'Cautious with purchases',
            'Willing to invest in premium products',
            'Prioritizes essential purchases',
            'Spends despite financial constraints',
            'Moderate spending habits'
        ]
    }
    customer_profile_df = pd.DataFrame(customer_profile)
    st.table(customer_profile_df)
    st.write("By understanding customer profile, we can plan suitable target marketing. Go to Prediction Page for details.")

    st.subheader("2. Final Cluster Result")
    image_cluster = Image.open("final cluster result.png")
    st.image(image_cluster,use_container_width=True)
    st.write("This result indicated that we need more data samples to ensure some age groups are well-represented in each cluster.")
else:
    st.header("Prediction Page")

    # Load the preprocessor and model with a context manager
    with open("preprocess.joblib", "rb") as file:
        preprocessor = joblib.load(file)

    with open("kmeans-best-model.joblib", "rb") as file2:
        kmeans_model = joblib.load(file2)
    
    # Dataset loading and overview
    data = pd.read_csv("final_dataset.csv")
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    # Get numeric features
    numeric_features = data.select_dtypes(include=np.number)
    numeric_feature_range = numeric_features.agg(['min', 'max'])

    # Get categorical features
    categorical_features = data.select_dtypes(exclude=np.number)
    categorical_feature_groups = categorical_features.apply(lambda x: x.unique().tolist())

    # UI for Numeric Features
    st.subheader("Input annual income (thousand dollars) and spending score (1-100)")
    st.write("Adjust the range for inputs using sliders:")
    numeric_inputs = {}
    for feature in numeric_features.columns:
        min_value = numeric_feature_range.loc['min', feature]
        max_value = numeric_feature_range.loc['max', feature]
        numeric_inputs[feature] = st.slider(
            label=f"{feature}",
            min_value=float(min_value),
            max_value=float(max_value)+1,
            value=float((min_value + max_value) / 2)
        )

    # UI for Categorical Features
    st.subheader("Input age group category")
    st.write("Please choose the age group category of the customer. Please refer to the Segmentation Result for the classification of age based on age group.")
    categorical_inputs = {}
    for feature in categorical_features.columns:
        options = categorical_feature_groups[feature]
        categorical_inputs[feature] = st.selectbox(
            label=f"{feature}",
            options=options
        )
    
    # Combine the numeric and categorical inputs
    user_inputs = {}

    # Add numeric inputs
    for feature, selected_range in numeric_inputs.items():
        # Take the midpoint of the range for simplicity
        user_inputs[feature] = np.mean(selected_range)

    # Add categorical inputs
    for feature, selected_value in categorical_inputs.items():
        user_inputs[feature] = selected_value

    # Create a DataFrame from user inputs
    user_input_df = pd.DataFrame([user_inputs])

    # Reorder the DataFrame columns to match X_train_log_modified
    user_input_df = user_input_df[data.columns]

    # Display the combined and ordered DataFrame
    st.header("Combined User Input Data")
    st.write(user_input_df)

        # Preprocess user input
    if preprocessor is not None:
        user_input_processed = preprocessor.transform(user_input_df)
    else:
        st.warning("Preprocessor is not available.")

    # marketing intiative dictionary
    marketing_strategy = {0: 'Customer has high annual income but low spending. Recommended marketing approaches are personalized offers, loyalty program through points for repeat purchases, premium branding.'
                          ,1: 'Senior customer with high annual income and high spending.  Upselling or cross-selling by introducing complementary products to what they are purchasing, Establishing VIP Events to build a sense of community and loyalty while showcasing premium offerings, and promoting high-end products suitable for their lifestyle are advisable.'
                          ,2: 'Customer has low annual income and spending. Consider value promotions, bundling offers, and referral programs to encourage word-of-mouth for increasing sales.'
                          ,3: 'Customer is part of young, productive population with low annual income but high spending. Try FOMO and trends (limited-time offers to create sense of urgency), engagement via influencers, emphasize experience.'
                          ,4: 'Customer is a mediocre spender and earner. Utilize targeted campaigns for previously popular products, reward programs through cashback and point, lifestyle-related product promotions.'
                          }

    # Predict the customer profile and recommended marketing approach
    if kmeans_model is not None:
        st.header("Predicted Customer Profile and Marketing Approach Suggestions")
        try:
            prediction_result = kmeans_model.predict(user_input_processed)[0]
            # Print the corresponding marketing strategy
            if prediction_result in marketing_strategy:
                st.success(f"Prediction result: {marketing_strategy[prediction_result]}")
            else:
                st.write("No marketing strategy found for the given prediction.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please upload a model to make predictions.")