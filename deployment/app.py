import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PIL import Image
import joblib
# from customer_segment_eda import eda_result
# from customer_segment_pred import pred_result

# Navigation
st.title("Mall Customer Segmentation and Recommendation Strategies")
page = st.selectbox("Select a page", ["Home", "EDA","About the Model","Go Predict!"])

if page == "Home":
    st.header("Welcome to the Home Page")
    st.write("The goal of this project is to give marketing recommendation strategies based on the mall customer's profile. We will use clustering algorithm to group customers with similar characteristics.")

    
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

elif page == "About the Model":
    st.header("About the Model")
    st.write("In this study, three clustering algorithms were chosen: K-Means, DBSCAN, and Gaussian Mixture Model. Then, the dataset were modified into four combinations and compared with the control group. Below is the details of the experimental content.")
    data_categories = {'data_number':['1','2','3','4','control'],
                  'data_content':['spending score, annual income, age, gender','spending score, annual income, age','spending score, annual income, age_group, gender','spending score, annual income, age_group','spending score, annual income']}
    st.table(data_categories)
    st.write("Based on the Silhouette Score and Davies-Bouldin index, K-Means model trained with data content from experiment 4 performed well due to comparable cluster quality with and richer explanation than control group.")

else:
    st.header("Prediction Page")
