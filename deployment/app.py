import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from customer_segment_eda import eda_result
# from customer_segment_pred import pred_result

# Navigation
st.title("App Navigation")
page = st.selectbox("Select a page", ["Home", "EDA Page","Prediction Page"])

if page == "Home":
    st.write("Welcome to the Home Page!")
    
elif page == "EDA Page":
    st.header("Exploratory Data Analysis")


else:
    st.header("Prediction Page")