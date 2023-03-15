import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('dat.csv')

ind_col = [col for col in data.columns if col != 'class']
dep_col = 'class'
X = data[ind_col]
y = data[dep_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rd_clf.fit(X_train, y_train)

def chronic_kid_Analysis():
    
    st.write('Dataset Size')
    st.info(data.shape)
    st.write('Dataset Columns')
    st.info(data.columns)
    st.write('Dataset Sample')
    st.write(data.head())
    
    st.write('Shape of our Training data')
    st.info(X_train.shape)
    st.write('Shape of our Test data')
    st.info(X_test.shape)

    #st.subheader('Applying Variance Threshold method of Feature Selection')

    #from sklearn.feature_selection import VarianceThreshold
    #var_thres = VarianceThreshold(threshold=0.0)#to drop columns of 10% variance
    #var_thres.fit(X_train)

    #st.write('Number of features that does not have ZERO Variance')
    #st.info(sum(var_thres.get_support()))

    #st.write('Shape of our Training data')
    #st.info(X_train.shape)
    #st.write('Shape of our Test data')
    #st.info(X_test.shape)

    st.subheader('Fitting our best model : Random Forest Classifier')

    

    # accuracy score, confusion matrix and classification report of random forest

    rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))
    st.write('Accuracy Score')
    st.info(rd_clf_acc)

    st.write('Classification Report')
    st.info(classification_report(y_test, rd_clf.predict(X_test)))


def chronic_predict():
    age = st.slider("Enter your age", 0, 100, 1)
    bp = st.slider("Enter your Blood Pressure",50,150,5)
    specific_gravity = st.selectbox("specific_gravity", [1.02 , 1.01 , 1.005, 1.015, 1.025])
    albumin = st.slider("Enter your albumin", 0, 10, 1)	
    sugar = st.slider("Enter your Sugar level", 0, 5, 1)	
    red_blood_cells = st.selectbox("Country", [0,1])
    pus_cell = st.selectbox("pus_cell", [0,1])
    pus_cell_clumps = st.selectbox("pus_cell_clumps", [0,1])
    bacteria = st.selectbox("bacteria", [0,1])
    blood_glucose_random = st.slider("Enter your blood_glucose_random level", 20, 500, 3)
    blood_urea = st.slider("Enter your blood_urea level", 1, 400, 3)
    serum_creatinine = st.slider("Enter your serum_creatinine level", 0, 76, 1)
    sodium = st.slider("Enter your sodium level", 4, 164, 10)
    potassium = st.slider("Enter your potassium level", 1, 50, 1) 
    haemoglobin = st.slider("Enter your haemoglobin level", 3, 18, 1) 
    packed_cell_volume = st.slider("Enter your packed_cell_volume level", 9, 54, 1) 
    white_blood_cell_count = st.slider("Enter your white_blood_cell_count level", 2200, 26400, 100) 
    red_blood_cell_count = st.slider("Enter your red_blood_cell_count level", 2, 8, 1) 
    hypertension = st.selectbox("hypertension", [0,1])
    diabetes_mellitus = st.selectbox("diabetes_mellitus", [0,1])
    coronary_artery_disease = st.selectbox("coronary_artery_disease", [0,1])
    appetite = st.selectbox("appetite", [0,1])
    peda_edema = st.selectbox("peda_edema", [0,1])
    aanemia	= st.selectbox("aanemia", [0,1])

    ok = st.button("Predict")
    if ok:
        pred = rd_clf.predict([[age,bp,specific_gravity,albumin,sugar,red_blood_cells,pus_cell,pus_cell_clumps,bacteria,blood_glucose_random,blood_urea,serum_creatinine,sodium,potassium,haemoglobin,packed_cell_volume,white_blood_cell_count,red_blood_cell_count,hypertension,diabetes_mellitus,coronary_artery_disease,appetite,peda_edema,aanemia]])
        st.write(pred)
        
        if pred == 0:
            st.write("You don't have any Chronic kidney Disease")
        else:
            st.write("You have Chronic kidney Disease")
    