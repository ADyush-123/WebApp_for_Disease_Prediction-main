from chronic_kidney_page import chronic_kid_Analysis
from chronic_kidney_page import chronic_predict
import streamlit as st


def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-photo/young-handsome-physician-medical-robe-with-stethoscope_1303-17818.jpg?w=996");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()
page = st.sidebar.selectbox("Explore Or Predict", ("Chronic Kidney Prediction", "Chronic Kidney Exploration","Coronary Artery Prediction", "Coronary Artery Exploration"))

#st.set_page_config(page_title='Disease Prediction',layout='wide')
st.title("Disease Prediction App")
#analysis = st.button("Click this button for Chronic kidney disease data Analysis")
#if analysis:
#    chronic_kid_Analysis()
#okk = st.button("Click this button for Chronic kidney prediction")
#if okk:
#    chronic_predict()

#okkk = st.button("Click this button for Coronary Artey disese prediction")
#if okkk:
#    pass
#
#analysiss = st.button("Click this button for Coronary Artery disease data Analysis")
#if analysiss:
#    pass



if page == "Chronic Kidney Prediction":
    chronic_predict()
elif page == 'Chronic Kidney Exploration':
    chronic_kid_Analysis()
elif page == 'Coronary Artery Prediction':
    pass
else:
    pass
